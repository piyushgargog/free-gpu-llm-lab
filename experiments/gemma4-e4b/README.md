# Experiment 3 — Gemma 4 E4B (Q4_0 GGUF, llama.cpp)

**Status: PARTIALLY COMPLETE. CUDA build was reached; a full generation
benchmark was NOT established in the original experiment. Do not treat any
Gemma 4 tokens/sec figure as observed — none exists.**

## Model

Gemma 4 E4B IT — approximately 8B total parameters, ~4.5B effective
parameters. Native BF16 is approximately ~16 GB, which is not a comfortable
fit on a free T4 (~15 GB VRAM). This experiment therefore uses the official
**Q4_0 GGUF** quantization instead.

- Repository: `google/gemma-4-E4B-it-qat-q4_0-gguf`
- File: `gemma-4-E4B_q4_0-it.gguf`
- Approximate download size: **~5.15 GB**

This file must **never** be downloaded on a personal PC — see
`download.py` below and `docs/colab.md`.

## llama.cpp / llama-cpp-python path

The initial `llama-cpp-python` install was CPU-only:

```python
from llama_cpp import llama_cpp
print(llama_cpp.llama_supports_gpu_offload())  # False
```

A CUDA build was attempted remotely:

```bash
pip uninstall -y llama-cpp-python

CMAKE_ARGS="-DGGML_CUDA=on" \
CMAKE_BUILD_PARALLEL_LEVEL=2 \
pip install llama-cpp-python --no-cache-dir --force-reinstall
```

Then re-checked:

```python
from llama_cpp import llama_cpp
print(llama_cpp.llama_supports_gpu_offload())  # expected True after a successful CUDA build
```

## What is and isn't established

- ✅ Q4_0 GGUF identified as the right fit for a free T4.
- ✅ CPU-only `llama-cpp-python` install confirmed via `llama_supports_gpu_offload() == False`.
- ✅ CUDA build procedure documented (`CMAKE_ARGS="-DGGML_CUDA=on"`).
- ❌ **A completed, benchmarked Gemma 4 CUDA inference run was NOT established** in
  the original experiment. No tokens/sec number exists for Gemma 4 E4B in this repo.

The llama.cpp CUDA investigation from this experiment directly carried
forward into the next experiment (`experiments/muse-glimmer-30b`), which
*does* have completed, benchmarked results using a from-source llama.cpp
CUDA build.

## Files

- `download.py` — remote-only download script for the Q4_0 GGUF file (guarded; requires `--confirm-remote-gpu`).
- `check_cuda.py` — checks `llama_supports_gpu_offload()`. Safe to run anywhere; reports `False` without a CUDA build/GPU, which is expected on a personal PC.
- `run.py` — llama.cpp inference wrapper. Remote GPU only. **NOT VERIFIED with a completed benchmark.**
- `benchmark.py` — benchmark harness (same shape as `experiments/muse-glimmer-30b/benchmark.py`) for whenever this experiment is picked back up. **Not yet run.**

## Reproducibility status

IN PROGRESS. To complete this experiment: build llama.cpp with CUDA (or
install a prebuilt CUDA wheel) in Colab/Kaggle, download the Q4_0 GGUF,
run `benchmark.py`, and record a real result.
