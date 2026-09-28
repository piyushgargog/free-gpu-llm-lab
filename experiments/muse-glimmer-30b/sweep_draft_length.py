#!/usr/bin/env python
"""Controlled sweep over --spec-draft-n-max for Muse Glimmer DFlash.

*** REMOTE GPU ONLY (Kaggle, 2x Tesla T4). NOT YET RUN from this repo --
no remote execution is available from the local dev environment. Run this
on Kaggle and its results will append real rows to results.jsonl. ***

Runs the SAME benchmark (model, draft model, hardware, prompt, context,
n_predict, -ngl) at each candidate --spec-draft-n-max value, changing only
that one variable, per optimize.md's controlled-optimization methodology.

Candidate values default to 8, 10, 12, 14, 15. This is the drafter's
ENTIRE valid --spec-draft-n-max range: the DFlash drafter's trained
block size is externally verified as dflash.block_size=16 (see
optimize.md), and llama.cpp's common/speculative.cpp clamps
--spec-draft-n-max to block_size-1=15 for standard (non-anchor) DFlash.
15 is therefore a confirmed hard ceiling, not merely a value that
happened to work. Values above 15 are refused unless --allow-above-15
is passed explicitly -- that flag should not be needed for this drafter;
it exists only in case a different DFlash drafter file (with a different
block_size) is ever substituted in.

If the best candidate in the sweep beats the current recorded best
(17.9 tok/s @ n_max=15), it is re-run once more (identical config) before
being reported as a verified new best, per the experiment protocol.
Crashing configurations are recorded as FAILED (valid=false), not skipped
silently.

Usage:
    python experiments/muse-glimmer-30b/sweep_draft_length.py \\
        --binary /kaggle/working/llama.cpp/build/bin/llama-cli \\
        --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \\
        --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
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

CURRENT_BEST_TOKENS_PER_SECOND = 17.9
CURRENT_BEST_N_MAX = 15
MAX_VALID_N_MAX = 15  # externally verified: dflash.block_size=16, so block_size-1=15 (see optimize.md)


def run_once(binary: str, model: str, draft: str, n_max: int, n_gpu_layers: int, context: int,
             n_predict: int, prompt: str) -> dict:
    cmd = build_command(binary, model, draft, n_gpu_layers, context, n_predict, prompt, None, False, n_max)
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except OSError as exc:
        elapsed = time.perf_counter() - start
        return {"status": "FAILED", "returncode": None, "output": "", "elapsed": elapsed,
                "reason": f"could not launch llama-cli binary ({exc})"}
    elapsed = time.perf_counter() - start
    combined = proc.stdout + "\n" + proc.stderr

    if proc.returncode != 0:
        return {"status": "FAILED", "returncode": proc.returncode, "output": combined, "elapsed": elapsed,
                "reason": f"llama-cli exited with code {proc.returncode}"}

    timings = parse_llama_cpp_timings(combined)
    if timings["eval_tokens_per_second"] is None:
        return {"status": "FAILED", "returncode": proc.returncode, "output": combined, "elapsed": elapsed,
                "reason": "could not parse generation tok/s from output"}

    return {"status": "OK", "timings": timings, "elapsed": elapsed, "output": combined}


def record(output_path: str, hardware: str, model: str, n_max: int, draft: str, context: int,
           n_predict: int, prompt: str, run_result: dict, run_label: str) -> None:
    if run_result["status"] == "FAILED":
        result = BenchmarkResult(
            model="Muse-Glimmer-30B-Q4_K_M",
            hardware=hardware,
            model_path=model,
            tokens_per_second=0.0,
            quantization="Q4_K_M",
            context_length=context,
            speculative_decoding=f"DFlash draft={draft}, spec-draft-n-max={n_max}",
            prompt=prompt,
            generated_tokens=n_predict,
            elapsed_seconds=run_result["elapsed"],
            git_commit=get_git_commit(),
            valid=False,
            notes=f"FAILED draft-length sweep {run_label} (spec-draft-n-max={n_max}): {run_result['reason']}",
        )
    else:
        timings = run_result["timings"]
        result = BenchmarkResult(
            model="Muse-Glimmer-30B-Q4_K_M",
            hardware=hardware,
            model_path=model,
            tokens_per_second=timings["eval_tokens_per_second"],
            prompt_tokens_per_second=timings["prompt_tokens_per_second"],
            quantization="Q4_K_M",
            context_length=context,
            speculative_decoding=f"DFlash draft={draft}, spec-draft-n-max={n_max}",
            prompt=prompt,
            generated_tokens=n_predict,
            elapsed_seconds=run_result["elapsed"],
            git_commit=get_git_commit(),
            valid=True,
            notes=(
                f"Draft-length sweep {run_label} (spec-draft-n-max={n_max}). "
                "Acceptance/draft statistics not available: llama-cli does not print them "
                "(only llama-server does, per llama.cpp docs/speculative.md)."
            ),
        )
    append_result(result, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument(
        "--values", default="8,10,12,14,15",
        help="Comma-separated --spec-draft-n-max candidates to test. All must be "
             f"<= {MAX_VALID_N_MAX} unless --allow-above-15 is passed.",
    )
    parser.add_argument("--n-gpu-layers", type=int, default=99)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--hardware", default="2x Tesla T4")
    parser.add_argument("--output", default="experiments/muse-glimmer-30b/results.jsonl")
    parser.add_argument(
        "--allow-above-15", action="store_true",
        help="Required to test any --spec-draft-n-max value > 15. For this drafter "
             "(externally verified dflash.block_size=16, see optimize.md), 15 is the "
             "confirmed maximum valid value -- llama.cpp will silently clamp anything "
             "higher back down to 15, so this flag should not be needed. It exists only "
             "for a future, different DFlash drafter file with a different block_size.",
    )
    args = parser.parse_args()

    values = [int(v.strip()) for v in args.values.split(",") if v.strip()]
    if not args.allow_above_15:
        unsafe = [v for v in values if v > MAX_VALID_N_MAX]
        if unsafe:
            print(
                f"Refusing to run: {unsafe} exceed the confirmed maximum valid value of "
                f"{MAX_VALID_N_MAX} (externally verified dflash.block_size=16 -- see "
                "optimize.md). If you are using a different DFlash drafter file, verify "
                "its own block_size first, then re-run with --allow-above-15.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"{'n_max':>6} | {'status':>7} | {'gen tok/s':>10} | {'prompt tok/s':>12}")
    ok_results: dict[int, dict] = {}
    for n_max in values:
        r = run_once(args.binary, args.model, args.draft, n_max, args.n_gpu_layers, args.context,
                      args.n_predict, args.prompt)
        record(args.output, args.hardware, args.model, n_max, args.draft, args.context,
               args.n_predict, args.prompt, r, "run 1")

        if r["status"] == "FAILED":
            print(f"{n_max:>6} | {'FAILED':>7} | {'-':>10} | {'-':>12}  ({r['reason']})")
            continue

        gen_tps = r["timings"]["eval_tokens_per_second"]
        prompt_tps = r["timings"]["prompt_tokens_per_second"]
        ok_results[n_max] = r
        print(f"{n_max:>6} | {'OK':>7} | {gen_tps:>10.2f} | {str(prompt_tps):>12}")

    if not ok_results:
        print(f"\nAll sweep runs FAILED. Current best remains {CURRENT_BEST_TOKENS_PER_SECOND} tok/s "
              f"(spec-draft-n-max={CURRENT_BEST_N_MAX}).")
        return

    best_n_max = max(ok_results, key=lambda k: ok_results[k]["timings"]["eval_tokens_per_second"])
    best_tps = ok_results[best_n_max]["timings"]["eval_tokens_per_second"]

    print(f"\nBest in this sweep: spec-draft-n-max={best_n_max} at {best_tps:.2f} tok/s")
    print(f"Current recorded best: spec-draft-n-max={CURRENT_BEST_N_MAX} at {CURRENT_BEST_TOKENS_PER_SECOND} tok/s")

    if best_tps <= CURRENT_BEST_TOKENS_PER_SECOND:
        print("Sweep did NOT beat the current best. 17.9 tok/s (spec-draft-n-max=15) remains the verified best.")
        return

    print(f"\nspec-draft-n-max={best_n_max} beat the current best in this sweep "
          f"({best_tps:.2f} > {CURRENT_BEST_TOKENS_PER_SECOND}). Re-running once more to verify "
          "before declaring a new best...")
    verify = run_once(args.binary, args.model, args.draft, best_n_max, args.n_gpu_layers, args.context,
                       args.n_predict, args.prompt)
    record(args.output, args.hardware, args.model, best_n_max, args.draft, args.context,
           args.n_predict, args.prompt, verify, "verification run 2")

    if verify["status"] == "FAILED":
        print(f"Verification run FAILED ({verify['reason']}). NOT declaring a new best. "
              f"Current best remains {CURRENT_BEST_TOKENS_PER_SECOND} tok/s (spec-draft-n-max={CURRENT_BEST_N_MAX}).")
        return

    verify_tps = verify["timings"]["eval_tokens_per_second"]
    print(f"Verification run: {verify_tps:.2f} tok/s")

    if verify_tps > CURRENT_BEST_TOKENS_PER_SECOND:
        print(f"\nVERIFIED NEW BEST: spec-draft-n-max={best_n_max}, run1={best_tps:.2f} tok/s, "
              f"run2={verify_tps:.2f} tok/s. Update README.md/optimize.md's recorded best to this value.")
    else:
        print(f"\nVerification run did not reproduce the improvement ({verify_tps:.2f} <= "
              f"{CURRENT_BEST_TOKENS_PER_SECOND}). NOT declaring a new best -- treating run 1 as noise. "
              f"Current best remains {CURRENT_BEST_TOKENS_PER_SECOND} tok/s (spec-draft-n-max={CURRENT_BEST_N_MAX}).")


if __name__ == "__main__":
    main()
