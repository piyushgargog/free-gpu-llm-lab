#!/usr/bin/env python
"""Thin wrapper around the llama.cpp `llama-cli` binary for Muse Glimmer 30B.

*** REMOTE GPU ONLY (Kaggle, 2x Tesla T4, llama.cpp built per README.md). ***

This script spawns the `llama-cli` binary as a subprocess — it does not
load anything into this Python process itself, but the subprocess it
launches DOES load multi-GB model weights onto a GPU. Never run on a
personal PC (the binary won't exist there anyway).

Usage:
    python experiments/muse-glimmer-30b/run.py \\
        --binary /kaggle/working/llama.cpp/build/bin/llama-cli \\
        --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \\
        --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf \\
        --dflash --spec-draft-n-max 15
"""

from __future__ import annotations

import argparse
import subprocess
import sys

DEFAULT_PROMPT = (
    "Explain what a transformer neural network is in simple technical terms. "
    "Give a concise technical explanation."
)


def build_command(
    binary: str,
    model: str,
    draft: str | None,
    n_gpu_layers: int,
    context: int,
    n_predict: int,
    prompt: str,
    batch_size: int | None,
    flash_attention: bool,
    spec_draft_n_max: int,
) -> list[str]:
    cmd = [binary, "-m", model, "-ngl", str(n_gpu_layers), "-c", str(context), "-n", str(n_predict),
           "-p", prompt, "--no-display-prompt"]
    if draft:
        cmd += ["-md", draft, "--spec-type", "draft-dflash", "--spec-draft-n-max", str(spec_draft_n_max)]
    if batch_size:
        cmd += ["-b", str(batch_size)]
    if flash_attention:
        cmd += ["-fa", "on"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True, help="Path to the llama-cli binary")
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", default=None, help="DFlash drafter GGUF path; omit to run without speculative decoding")
    parser.add_argument("--spec-draft-n-max", type=int, default=15)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--flash-attention", action="store_true")
    args = parser.parse_args()

    cmd = build_command(
        args.binary, args.model, args.draft, args.n_gpu_layers, args.context, args.n_predict,
        args.prompt, args.batch_size, args.flash_attention, args.spec_draft_n_max,
    )
    print("Running:", " ".join(cmd), file=sys.stderr)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
