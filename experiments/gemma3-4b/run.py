#!/usr/bin/env python
"""Load Gemma 3 4B and generate text, with selectable dtype.

*** REMOTE GPU ONLY (Colab/Kaggle T4). Never run on a personal PC. ***

Usage:
    python experiments/gemma3-4b/run.py --dtype bf16 --prompt "..."
    python experiments/gemma3-4b/run.py --dtype nf4  --prompt "..."
    python experiments/gemma3-4b/run.py --dtype fp16 --prompt "..."   # reproduces the NaN-logit failure

--dtype fp16 is included ONLY to reproduce/document the historical
numerical-stability failure (see README.md). Its output must never be
treated as a valid benchmark.
"""

from __future__ import annotations

import argparse

from free_gpu_llm.config import GenerationConfig, ModelConfig, get_hf_token
from free_gpu_llm.generation import build_messages, generate, load_model_and_tokenizer

REPO_ID = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are Gemma 3 4B, an AI assistant running locally on a Google Colab "
    "NVIDIA T4 GPU. You are not Gemini. Answer accurately and concisely. "
    "If you don't know something, say so rather than inventing information."
)

_DTYPE_MAP = {"fp16": "float16", "bf16": "bfloat16", "nf4": "nf4"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtype", choices=list(_DTYPE_MAP), default="bf16")
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms.")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    args = parser.parse_args()

    model_config = ModelConfig(
        name=f"gemma-3-4b-it-{args.dtype}",
        repo_id=REPO_ID,
        dtype=_DTYPE_MAP[args.dtype],
        attn_implementation="sdpa",
    )
    model, tokenizer = load_model_and_tokenizer(model_config, hf_token=get_hf_token())

    messages = build_messages(SYSTEM_PROMPT, history=[], user_message=args.prompt)
    gen_config = GenerationConfig(max_new_tokens=args.max_new_tokens)
    output = generate(model, tokenizer, messages, gen_config)

    if args.dtype == "fp16":
        print("WARNING: fp16 is known to produce NaN logits for this model on T4. "
              "Output below is NOT a valid benchmark. See README.md.\n")
    print(output)


if __name__ == "__main__":
    main()
