#!/usr/bin/env python
"""Run llama-cli and record a Muse Glimmer benchmark result.

*** REMOTE GPU ONLY. *** Spawns the llama-cli binary (subprocess), captures
its timing output, parses tok/s, and appends a BenchmarkResult to
results.jsonl. Follow the controlled-optimization methodology in
optimize.md: change ONE variable per run, compare against the current best
(17.9 tok/s DFlash), keep only demonstrated improvements.

Usage:
    python experiments/muse-glimmer-30b/benchmark.py \\
        --binary /kaggle/working/llama.cpp/build/bin/llama-cli \\
        --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \\
        --notes "baseline"

    python experiments/muse-glimmer-30b/benchmark.py \\
        --binary /kaggle/working/llama.cpp/build/bin/llama-cli \\
        --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \\
        --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \\
        --notes "DFlash spec-draft-n-max=15"
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import DEFAULT_PROMPT, build_command  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from free_gpu_llm.benchmarking import (  # noqa: E402
    BenchmarkResult,
    append_result,
    get_git_commit,
    parse_llama_cpp_timings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", default=None)
    parser.add_argument("--spec-draft-n-max", type=int, default=15)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--hardware", default="2x Tesla T4")
    parser.add_argument("--notes", required=True, help="What is being tested, e.g. 'baseline' or 'batch 512'")
    parser.add_argument("--output", default="experiments/muse-glimmer-30b/results.jsonl")
    args = parser.parse_args()

    cmd = build_command(
        args.binary, args.model, args.draft, args.n_gpu_layers, args.context, args.n_predict,
        args.prompt, args.batch_size, args.flash_attention, args.spec_draft_n_max,
    )

    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        print(f"Could not launch llama-cli binary at {args.binary!r}: {exc}", file=sys.stderr)
        sys.exit(1)
    elapsed = time.perf_counter() - start
    combined_output = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        print(f"llama-cli exited with code {proc.returncode}", file=sys.stderr)
        print(combined_output, file=sys.stderr)
        sys.exit(proc.returncode)

    timings = parse_llama_cpp_timings(combined_output)
    if timings["eval_tokens_per_second"] is None:
        print("Could not parse generation tok/s from llama-cli output.", file=sys.stderr)
        print(combined_output, file=sys.stderr)
        sys.exit(1)

    result = BenchmarkResult(
        model="Muse-Glimmer-30B-Q4_K_M",
        hardware=args.hardware,
        model_path=args.model,
        tokens_per_second=timings["eval_tokens_per_second"],
        prompt_tokens_per_second=timings["prompt_tokens_per_second"],
        quantization="Q4_K_M",
        context_length=args.context,
        batch_size=args.batch_size,
        speculative_decoding=(f"DFlash draft={args.draft}, spec-draft-n-max={args.spec_draft_n_max}" if args.draft else None),
        prompt=args.prompt,
        generated_tokens=args.n_predict,
        elapsed_seconds=elapsed,
        git_commit=get_git_commit(),
        valid=True,
        notes=args.notes,
    )
    append_result(result, args.output)
    print(f"generation: {result.tokens_per_second:.2f} tok/s | prompt: {result.prompt_tokens_per_second} tok/s -> recorded to {args.output}")


if __name__ == "__main__":
    main()
