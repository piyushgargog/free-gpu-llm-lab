# Troubleshooting

Real issues actually hit during these experiments, and how they were
resolved (or where they still stand). See `docs/experiments.md` for the
full narrative per experiment.

## GPTQ Marlin kernel fails on Tesla T4 (Qwen3-30B-A3B)

**Symptom:** GPTQ model load/inference fails or errors relating to the
Marlin kernel / JIT compilation on T4.

**Cause:** Marlin targets newer GPU architectures than T4 (compute
capability 7.5).

**Fix:** Force the plain PyTorch dequant backend: `GPTQConfig(backend="torch")`.
Much slower (~1.8 tok/s observed for a 30B MoE model on 2×T4 with a manual
layer split), but functional.

## `device_map="auto"` OOMs / places layers badly on multi-GPU

**Symptom:** GPU memory errors or a lopsided GPU memory split with
`device_map="auto"` across 2× T4.

**Fix used:** Manually build a `device_map` dict assigning specific layers
to specific GPU indices (see `experiments/qwen3-30b-a3b/run.py:build_manual_device_map`).
This is a blunt tool — it assumes you know (or can inspect) the model's
layer count and naming convention — but it sidesteps `device_map="auto"`'s
heuristics when they get it wrong for a given model/hardware combination.

## FP16 produces NaN logits (Gemma 3 4B)

**Symptom:** Model loads and "generates" without raising an exception, but
output looks garbled/repetitive or is mostly padding tokens.

**Cause:** Numerical overflow under FP16's narrower exponent range for this
model's activation distribution on this hardware.

**Fix:** Switch to `dtype=torch.bfloat16`. Always validate logits
(`free_gpu_llm.generation.validate_logits`) rather than trusting that
"it generated something" means it worked — see `docs/numerical-stability.md`.

## Memory-efficient attention kernel fails to engage (Gemma 3 4B)

**Symptom:** Certain attention backends silently fall back or fail for
Gemma 3 4B.

**Cause:** Gemma 3 4B uses grouped-query attention with 8 query heads and
4 KV heads (unequal counts); some attention kernel implementations assume
equal Q/KV head counts.

**Fix used:** `attn_implementation="sdpa"`, which handled the GQA head
config correctly. Unsupported kernels were not forced to "work" — this is
documented as a genuine compatibility limitation, not worked around with a
fabricated success claim.

## `llama-cpp-python` installs CPU-only by default

**Symptom:** `llama_cpp.llama_supports_gpu_offload()` returns `False` even
though a GPU is present.

**Cause:** The default pip wheel for `llama-cpp-python` is commonly built
without CUDA support.

**Fix:**

```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_CUDA=on" CMAKE_BUILD_PARALLEL_LEVEL=2 \
  pip install llama-cpp-python --no-cache-dir --force-reinstall
```

Then re-check `llama_supports_gpu_offload()` — see
`experiments/gemma4-e4b/check_cuda.py`.

## llama.cpp CMake build fails around `CUDA::cuda_driver`

**Symptom:** The default/native CMake configuration for building llama.cpp
from source fails to resolve the `CUDA::cuda_driver` target.

**Fix (observed working configuration):**

```bash
cmake -S /path/to/llama.cpp -B /path/to/llama.cpp/build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75
```

Explicitly setting `-DCMAKE_CUDA_ARCHITECTURES=75` (T4) and disabling
`GGML_NATIVE` avoids relying on architecture auto-detection, which appears
to be what triggered the original failure.

## DFlash prints "requires ctx_other to be set" at startup

**Symptom:** `dflash requires ctx_other to be set (this warning is normal
during memory fitting)` printed to console when starting a DFlash run.

**Status:** **Non-fatal.** Generation completes successfully afterward.
Documented here rather than hidden — if you see this and generation stalls
or errors, that's a *different* problem, not this warning.

## Batch size / Flash Attention don't improve Muse decode speed

**Not a bug** — documented finding, not a failure to fix. `-b 512` and
`-fa on` both left generation tok/s essentially unchanged (~13.6–13.7 tok/s
vs. baseline ~13.7 tok/s) while prompt-eval tok/s rose slightly. See
`experiments/muse-glimmer-30b/optimize.md` for why (decode throughput here
is likely bound by weight-loading/dequant bandwidth, not by batch
parallelism or attention compute).

## General: HF authentication failures

**Symptom:** `401`/gated-repo errors loading `google/gemma-3-4b-it` or
similar gated models.

**Fix:** Set `HF_TOKEN` (see `.env.example`) in the **remote** environment
and accept the model's license on huggingface.co with that account. Never
commit the token; `free_gpu_llm.config.get_hf_token()` reads it from the
environment only.
