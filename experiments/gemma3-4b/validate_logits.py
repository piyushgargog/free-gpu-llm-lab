#!/usr/bin/env python
"""Run one forward pass and validate the logits for NaN/Inf.

*** REMOTE GPU ONLY. ***

This is how the FP16 numerical-stability failure was originally caught:
FP16 appeared to load and generate, but logit validation revealed NaNs
and invalid token IDs. BF16 and NF4 validate clean (0 NaNs, 0 Infs).

Usage:
    python experiments/gemma3-4b/validate_logits.py --dtype fp16
    python experiments/gemma3-4b/validate_logits.py --dtype bf16
"""

from __future__ import annotations

import argparse

from free_gpu_llm.config import ModelConfig, get_hf_token
from free_gpu_llm.generation import load_model_and_tokenizer, validate_logits

REPO_ID = "google/gemma-3-4b-it"
_DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "nf4": "nf4"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms.")
    args = parser.parse_args()

    model_config = ModelConfig(
        name=f"gemma-3-4b-it-{args.dtype}",
        repo_id=REPO_ID,
        dtype=_DTYPE_MAP[args.dtype],
        attn_implementation="sdpa",
    )
    model, tokenizer = load_model_and_tokenizer(model_config, hf_token=get_hf_token())

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    import torch

    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    result = validate_logits(logits)
    print(f"dtype={args.dtype}")
    print(f"total values checked: {result.total}")
    print(f"NaN count: {result.nan_count}")
    print(f"Inf count: {result.inf_count}")
    print(f"VALID: {result.valid}")

    if not result.valid:
        print("\nThis configuration produces invalid logits and must NOT be reported "
              "as a valid benchmark (see docs/numerical-stability.md).")


if __name__ == "__main__":
    main()
