#!/usr/bin/env python
"""Run baseline vs. DFlash back-to-back and print a comparison.

*** REMOTE GPU ONLY. *** Runs two llama-cli invocations (baseline, then
DFlash) and reports the generation-speed delta, matching the observed
~13.7 -> ~17.9 tok/s (~30.7%) result documented in README.md.

Usage:
    python experiments/muse-glimmer-30b/dflash.py \\
        --binary /kaggle/working/llama.cpp/build/bin/llama-cli \\
        --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \\
        --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import DEFAULT_PROMPT, build_command  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from free_gpu_llm.benchmarking import parse_llama_cpp_timings  # noqa: E402


def run_once(binary: str, model: str, draft: str | None, args) -> dict:
    cmd = build_command(
        binary, model, draft, args.n_gpu_layers, args.context, args.n_predict,
        args.prompt, None, False, args.spec_draft_n_max,
    )
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        raise RuntimeError(f"could not launch llama-cli binary at {binary!r}: {exc}") from exc
    if proc.returncode != 0:
        print(proc.stdout, proc.stderr, file=sys.stderr)
        raise RuntimeError(f"llama-cli exited with code {proc.returncode}")
    return parse_llama_cpp_timings(proc.stdout + "\n" + proc.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--spec-draft-n-max", type=int, default=15)
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    print("Running baseline (no speculative decoding)...", file=sys.stderr)
    baseline = run_once(args.binary, args.model, None, args)

    print("Running DFlash...", file=sys.stderr)
    dflash = run_once(args.binary, args.model, args.draft, args)

    base_tps = baseline["eval_tokens_per_second"]
    dflash_tps = dflash["eval_tokens_per_second"]

    print(f"\nBaseline generation: {base_tps} tok/s")
    print(f"DFlash generation:   {dflash_tps} tok/s")

    if base_tps and dflash_tps:
        delta = dflash_tps - base_tps
        pct = (delta / base_tps) * 100
        print(f"Delta: {delta:+.2f} tok/s ({pct:+.1f}%)")
    else:
        print("Could not compute delta — one of the runs did not produce a parseable tok/s.")


if __name__ == "__main__":
    main()
