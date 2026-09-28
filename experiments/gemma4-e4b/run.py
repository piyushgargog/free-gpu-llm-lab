#!/usr/bin/env python
"""Run Gemma 4 E4B Q4_0 GGUF inference via llama-cpp-python.

*** REMOTE GPU ONLY. NOT VERIFIED — no completed benchmark exists for this
experiment (see README.md). Use experiments/muse-glimmer-30b for a
completed, benchmarked llama.cpp GGUF example. ***

Usage:
    python experiments/gemma4-e4b/run.py --model-path /path/to/gemma-4-E4B_q4_0-it.gguf --prompt "..."
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms.")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="99 offloads all layers to GPU")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    from llama_cpp import Llama

    llm = Llama(
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.context,
    )
    output = llm(args.prompt, max_tokens=args.max_tokens)
    print(output["choices"][0]["text"])


if __name__ == "__main__":
    main()
