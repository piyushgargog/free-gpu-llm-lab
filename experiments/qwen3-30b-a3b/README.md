# Experiment 1 — Qwen3-30B-A3B (GPTQ Int4, 2× T4)

**Status: HISTORICAL / NOT REPRODUCIBLE RIGHT NOW.** The model weights used
in this experiment were lost after a Kaggle session restart and are not
currently present in any environment. Numbers below are the observed
historical result; the loading code in this folder is a best-effort
reconstruction of the approach used and is marked **NOT VERIFIED** against
a live re-run.

## Environment

- Kaggle
- NVIDIA Tesla T4 × 2 (~15–16 GB VRAM each)

## Model

- `Qwen3-30B-A3B`, GPTQ Int4
- Mixture-of-Experts: ~30B total parameters, ~3B active parameters/token

## What went wrong (in order)

1. A single T4 could not practically fit the model.
2. `device_map="auto"` caused GPU memory placement problems.
3. Falling back to CPU/GPU splitting caused Kaggle host-RAM problems.
4. The GPTQ **Marlin** kernel hit a T4 JIT/backend compatibility issue
   (Marlin targets newer architectures than T4's compute capability 7.5).
5. Switched GPTQ backend to `"torch"` (slower, but compatible with T4).
6. Manually distributed transformer layers across the two T4s instead of
   relying on `device_map="auto"`.

## Observed historical configuration

| GPU | Layers |
|---|---|
| GPU 0 | ~28 |
| GPU 1 | ~20 |

## Observed historical performance

**~1.8 tokens/sec**

This is slow — consistent with `backend="torch"` GPTQ dequantization
(no fused Marlin kernel) plus cross-GPU activation transfer for the
manually split layers on T4 hardware with no NVLink.

## Files

- `run.py` — reconstructed manual multi-GPU GPTQ loading approach.
  **NOT VERIFIED** — the exact transformers/auto-gptq API surface should be
  re-checked against current package versions before relying on it; the
  original model is no longer available to re-test against.
- `benchmark.py` — generation timing harness, reusing
  `free_gpu_llm.benchmarking`. Remote GPU only.

## Reproducibility status

**NOT reproducible as-is** — the Qwen3-30B-A3B GPTQ weights that were used
are gone (lost on Kaggle session restart) and were never re-downloaded.
Re-running this experiment requires re-downloading the model in a fresh
Kaggle session (see `docs/colab.md`) and validating `run.py` against it.

## Lesson learned

GPTQ Marlin kernels are not guaranteed to work on Turing-generation GPUs
(T4). When a GPTQ model fails to load/run correctly on T4, try forcing the
plain PyTorch dequant backend before abandoning GPTQ entirely — at the cost
of significant throughput.
