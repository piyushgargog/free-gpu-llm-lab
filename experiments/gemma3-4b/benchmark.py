#!/usr/bin/env python
"""Timed generation benchmark for Gemma 3 4B, with logit validation.

*** REMOTE GPU ONLY. ***

Runs a forward pass first to validate logits (NaN/Inf) BEFORE trusting the
timing — a run with invalid logits is recorded with valid=False and its
tok/s is not to be treated as a real performance number, no matter what the
wall-clock timer says.

Historical observed results for this model (see README.md / results.jsonl):
  bf16: ~10.04 tok/s (valid)   |   nf4: ~7.84 tok/s (valid)   |   fp16: INVALID
"""

from __future__ import annotations

import argparse
import time

from free_gpu_llm.benchmarking import BenchmarkResult, append_result, get_git_commit, tokens_per_second
from free_gpu_llm.config import ModelConfig, get_hf_token
from free_gpu_llm.generation import build_messages, load_model_and_tokenizer, validate_logits

REPO_ID = "google/gemma-3-4b-it"
_DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "nf4": "nf4"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms. Give a concise technical explanation.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--output", default="experiments/gemma3-4b/results.jsonl")
    args = parser.parse_args()

    model_config = ModelConfig(
        name=f"gemma-3-4b-it-{args.dtype}",
        repo_id=REPO_ID,
        dtype=_DTYPE_MAP[args.dtype],
        attn_implementation="sdpa",
    )
    model, tokenizer = load_model_and_tokenizer(model_config, hf_token=get_hf_token())

    import torch

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        validation_pass = model(**inputs)
    validation = validate_logits(validation_pass.logits)

    messages = build_messages(None, history=[], user_message=args.prompt)
    chat_inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    start = time.perf_counter()
    output_ids = model.generate(**chat_inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    elapsed = time.perf_counter() - start

    generated_count = output_ids.shape[1] - chat_inputs["input_ids"].shape[1]

    result = BenchmarkResult(
        model=f"gemma-3-4b-it-{args.dtype}",
        hardware="1x Tesla T4",
        model_path=REPO_ID,
        tokens_per_second=tokens_per_second(generated_count, elapsed) if validation.valid else 0.0,
        dtype=_DTYPE_MAP[args.dtype],
        quantization="NF4" if args.dtype == "nf4" else None,
        generated_tokens=generated_count,
        elapsed_seconds=elapsed,
        prompt=args.prompt,
        git_commit=get_git_commit(),
        valid=validation.valid,
        notes=(
            "Valid." if validation.valid
            else f"INVALID: {validation.nan_count} NaNs / {validation.inf_count} Infs in logits — do not treat tok/s as real."
        ),
    )
    append_result(result, args.output)
    print(f"NaNs={validation.nan_count} Infs={validation.inf_count} valid={validation.valid}")
    print(f"{result.tokens_per_second:.2f} tok/s -> recorded to {args.output}")


if __name__ == "__main__":
    main()
