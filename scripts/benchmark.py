#!/usr/bin/env python
"""Record a benchmark result to a JSONL results file.

Two modes:
  1. --log-file: parse captured llama.cpp (llama-cli) stderr/stdout output
     for prompt/eval tokens-per-second.
  2. --tokens-per-second: supply the number directly (e.g. for a
     transformers-based benchmark that isn't llama.cpp).

This script only does text parsing and file I/O — it does not run any
model itself, so it's safe to invoke locally against a log file you
captured remotely and copied down (e.g. to add it to the repo's history).

Usage:
    python scripts/benchmark.py \\
        --model Muse-Glimmer-30B-Q4_K_M \\
        --hardware "2x Tesla T4" \\
        --quantization Q4_K_M --context 2048 \\
        --log-file muse_dflash_run.log \\
        --speculative-decoding "DFlash draft-dflash, spec-draft-n-max=15" \\
        --notes "Best observed Muse configuration" \\
        --output experiments/muse-glimmer-30b/results.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from free_gpu_llm.benchmarking import (  # noqa: E402
    BenchmarkResult,
    append_result,
    get_git_commit,
    parse_llama_cpp_timings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hardware", required=True, help='e.g. "2x Tesla T4"')
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--context", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--speculative-decoding", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--generated-tokens", type=int, default=None)
    parser.add_argument("--elapsed-seconds", type=float, default=None)
    parser.add_argument("--log-file", default=None, help="Captured llama.cpp output to parse tok/s from")
    parser.add_argument("--tokens-per-second", type=float, default=None, help="Provide directly instead of --log-file")
    parser.add_argument("--prompt-tokens-per-second", type=float, default=None)
    parser.add_argument("--software-version", default=None)
    parser.add_argument("--valid", dest="valid", action="store_true", default=True)
    parser.add_argument("--invalid", dest="valid", action="store_false", help="Mark this run as NOT a valid benchmark")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--output", required=True, help="Path to append this result to (JSONL)")
    args = parser.parse_args()

    eval_tps = args.tokens_per_second
    prompt_tps = args.prompt_tokens_per_second

    if args.log_file:
        text = Path(args.log_file).read_text(encoding="utf-8", errors="replace")
        parsed = parse_llama_cpp_timings(text)
        eval_tps = eval_tps if eval_tps is not None else parsed["eval_tokens_per_second"]
        prompt_tps = prompt_tps if prompt_tps is not None else parsed["prompt_tokens_per_second"]

    if eval_tps is None:
        print("Error: could not determine tokens_per_second (pass --tokens-per-second or a parseable --log-file)", file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult(
        model=args.model,
        hardware=args.hardware,
        model_path=args.model_path,
        tokens_per_second=eval_tps,
        prompt_tokens_per_second=prompt_tps,
        quantization=args.quantization,
        dtype=args.dtype,
        context_length=args.context,
        batch_size=args.batch_size,
        speculative_decoding=args.speculative_decoding,
        prompt=args.prompt,
        generated_tokens=args.generated_tokens,
        elapsed_seconds=args.elapsed_seconds,
        software_version=args.software_version,
        git_commit=get_git_commit(),
        valid=args.valid,
        notes=args.notes,
    )

    append_result(result, args.output)
    print(f"Recorded: {result.model} -> {result.tokens_per_second} tok/s (valid={result.valid}) -> {args.output}")


if __name__ == "__main__":
    main()
