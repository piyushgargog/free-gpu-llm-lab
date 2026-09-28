# Experiment Log

Chronological record of every experiment in this repo. Each entry: Goal,
Environment, Model, Configuration, Result, Failure (if any), Root cause,
Lesson, Reproducibility status.

---

## 1. Qwen3-30B-A3B (GPTQ Int4)

- **Goal:** Run a ~30B MoE model (~3B active/token) on free Kaggle GPUs.
- **Environment:** Kaggle, 2× Tesla T4.
- **Model:** Qwen3-30B-A3B, GPTQ Int4.
- **Configuration:** Manual layer split — GPU0: ~28 layers, GPU1: ~20 layers; GPTQ `backend="torch"`.
- **Result:** ~1.8 tokens/sec.
- **Failure:** `device_map="auto"` caused GPU memory problems; CPU/GPU split caused host-RAM problems; GPTQ **Marlin** kernel had a T4 JIT/backend compatibility issue.
- **Root cause:** Marlin targets newer GPU architectures than T4 (compute capability 7.5); falling back to the plain PyTorch dequant backend works but is much slower.
- **Lesson:** GPTQ Marlin is not guaranteed to work on Turing (T4) — check `backend="torch"` as a fallback before abandoning GPTQ.
- **Reproducibility status:** **NOT reproducible** — model weights lost after a Kaggle session restart; not re-downloaded.

## 2. Gemma 3 4B — FP16

- **Goal:** Establish a BF16/FP16 baseline for Gemma 3 4B on a single T4.
- **Environment:** Google Colab Free, 1× Tesla T4, CUDA 12.8, PyTorch 2.11.0+cu128, Transformers 5.16.1.
- **Model:** `google/gemma-3-4b-it`, `dtype=torch.float16`.
- **Result:** Appeared to load and generate (~8.6 GB VRAM), but logit validation revealed **NaN logits** and invalid/mostly-padding token IDs.
- **Failure:** Numerical instability under FP16.
- **Root cause:** Not root-caused further beyond "FP16 is numerically unstable for this model on this hardware" — see `docs/numerical-stability.md`.
- **Lesson:** Always validate logits (NaN/Inf) before trusting a benchmark's apparent success — a model can "generate text" while numerically broken.
- **Reproducibility status:** Reproducible (the failure itself), but its output is INVALID and must never be reported as a valid benchmark.

## 3. Gemma 3 4B — BF16

- **Goal:** Get a numerically valid baseline.
- **Environment:** Same as above.
- **Model:** `google/gemma-3-4b-it`, `dtype=torch.bfloat16`, `attn_implementation="sdpa"`.
- **Result:** ~8.03 GB VRAM; 0 NaNs, 0 Infs; 200 tokens in ~19.92s → **~10.04 tok/s**.
- **Failure:** None.
- **Lesson:** BF16 is the safe default for this model on T4.
- **Reproducibility status:** Valid, reproducible baseline.

## 4. Gemma 3 4B — NF4

- **Goal:** Reduce VRAM further via 4-bit quantization.
- **Environment:** Same as above, `bitsandbytes` NF4.
- **Result:** ~3.2 GB VRAM; 200 tokens in ~25.52s → **~7.84 tok/s**.
- **Failure:** None (valid, but slower than BF16).
- **Lesson:** NF4 trades ~60% less VRAM for ~22% less throughput in this specific setup — not a universal ratio.
- **Reproducibility status:** Valid, reproducible.

## 5. Gradio chatbot (Gemma 3 4B, no web)

- **Goal:** Interactive chat UI for the Colab-hosted model.
- **Configuration:** Model loaded once at startup; fixed system prompt; conversation history maintained.
- **Result:** Functional local conversational interface (see `experiments/gemma3-4b/gradio_app.py`).
- **Reproducibility status:** Code-reproducible; UI behavior not independently re-benchmarked.

## 6. Web-augmented Gemma (DDGS + trafilatura)

- **Goal:** Give the chatbot access to current information via external retrieval.
- **Configuration:** Keyword-triggered DDGS search → trafilatura page extraction → context injected into the prompt (`app/gradio_app.py`, `src/free_gpu_llm/web/`).
- **Result:** Functional retrieval pipeline. This is external retrieval + prompt augmentation — explicitly **not** autonomous browsing and **not** native model tool-calling.
- **Reproducibility status:** Code-reproducible; kept modular so it can later be replaced by real tool calling.

## 7. Gemma 4 E4B — llama.cpp CUDA investigation

- **Goal:** Fit a larger (~8B total / ~4.5B effective) Gemma 4 model onto a free T4 via GGUF quantization.
- **Model:** `google/gemma-4-E4B-it-qat-q4_0-gguf`, file `gemma-4-E4B_q4_0-it.gguf` (~5.15 GB).
- **Configuration:** Initial `llama-cpp-python` install was CPU-only (`llama_supports_gpu_offload() == False`); rebuilt with `CMAKE_ARGS="-DGGML_CUDA=on"`.
- **Result:** CUDA build procedure established. **A completed, benchmarked inference run was NOT established** in this experiment.
- **Failure:** Incomplete — no generation benchmark exists for Gemma 4 E4B.
- **Lesson:** Verify `llama_supports_gpu_offload()` before assuming a llama-cpp-python install has GPU support; the default pip wheel is commonly CPU-only.
- **Reproducibility status:** **IN PROGRESS.**

## 8. Muse Glimmer 30B — llama.cpp native CUDA baseline

- **Goal:** Run a ~28B GGUF model (Q4_K_M) across 2× T4 using a from-source CUDA build of llama.cpp (following on from the Gemma 4 llama.cpp investigation).
- **Environment:** Kaggle, 2× Tesla T4, llama.cpp `0.5.0-dev build 1 commit 81ef10e`, built with `-DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=75`.
- **Configuration:** `llama-cli -ngl 99 -c 2048 -n 256`.
- **Result:** Generation ~13.7 tok/s, prompt eval ~104.7 tok/s.
- **Failure:** The default CMake config failed around the `CUDA::cuda_driver` target; the config above is the fix.
- **Lesson:** A from-source CUDA build with explicit `CMAKE_CUDA_ARCHITECTURES=75` (T4's compute capability) is more reliable than relying on native/auto-detected architecture flags in this environment.
- **Reproducibility status:** Reproducible given the model files (remote only).

## 9. Muse batch-size experiment (`-b 512`)

- **Result:** Generation ~13.6 tok/s, prompt eval ~110.4 tok/s. **No useful decode-speed improvement.**
- **Lesson:** Batch size mainly helps prompt-processing parallelism, not single-stream autoregressive decode throughput.

## 10. Muse Flash Attention experiment (`-fa on`)

- **Result:** Generation ~13.7 tok/s, prompt eval ~109.7 tok/s. **No measurable decode-speed improvement.**
- **Lesson:** At this context length, decode throughput is likely bound by weight-loading/dequant bandwidth rather than attention compute.

## 11. Muse DFlash speculative decoding

- **Configuration:** `-md dflash-Muse-Glimmer-30B-Q4_K_M.gguf --spec-type draft-dflash --spec-draft-n-max 15`.
- **Result (our own measured benchmark):** Generation **~17.9 tok/s** — ~30.7% higher than the 13.7 tok/s baseline (~1.31×). Prompt tok/s not measured for this config.
- **Externally verified configuration fact (not our own measurement):**
  the drafter's `dflash.block_size = 16`, making `--spec-draft-n-max 15`
  the maximum value the implementation accepts (`block_size - 1`). This
  is a fact about the valid configuration range, not a performance
  finding — it does not mean 15 is the fastest value in that range.
- **Warning observed:** `dflash requires ctx_other to be set (this warning is normal during memory fitting)` — non-fatal; generation completed successfully.
- **Lesson:** Speculative decoding is a fundamentally different lever than kernel/batch tuning — it amortizes the big model's per-token cost by verifying multiple drafted tokens per forward pass.
- **Reproducibility status:** Reproducible given the model + drafter files (remote only). **Current best observed Muse configuration.**

## 12. Muse DFlash draft-length sweep (`--spec-draft-n-max`)

- **Goal:** Determine whether a `--spec-draft-n-max` value other than the
  first-tried 15 improves decode throughput beyond the 17.9 tok/s DFlash
  best.
- **Environment:** Same as §11 (Kaggle, 2× T4, same llama.cpp build).
- **Externally verified configuration fact (not our own measurement):**
  the drafter's `dflash.block_size = 16` (confirmed outside this repo's
  own artifacts — a dedicated forensic search of everything already in
  this repo could not recover it). Per `common/speculative.cpp`'s
  clamping logic, this makes `--spec-draft-n-max 15` the **maximum valid
  value** for standard DFlash (`block_size - 1`). `llama-cli` (used
  throughout this repo's benchmarks, unlike `llama-server`) is separately
  confirmed to **not** print draft acceptance statistics.
- **Configuration:** Identical to §11 (same model, draft, hardware,
  prompt, `-c 2048 -n 256`, `-ngl 99`, no `-b`/`-fa`), sweeping
  `--spec-draft-n-max` over `{8, 10, 12, 14, 15}` — this is now confirmed
  to be the **entire valid range** for this drafter, not merely a
  cautious subset of an unknown range. **Knowing 15 is the maximum valid
  value does not mean it is the fastest value** — that is exactly what
  this unrun sweep is for.
- **Result:** **NOT RUN.** No remote (Kaggle) execution is available from
  this dev environment. Harness (`experiments/muse-glimmer-30b/sweep_draft_length.py`)
  is built and ready; running it on Kaggle appends real results to
  `results.jsonl` and this log / the experiment README will be updated
  with the actual numbers.
- **Reproducibility status:** Harness ready; results pending remote execution.
  **17.9 tok/s (`spec-draft-n-max=15`) remains the current best** until a
  swept value is actually benchmarked and beats it.
- **Next experiment:** Once Kaggle is available, run the prepared sweep
  (`--spec-draft-n-max` ∈ `{8, 10, 12, 14, 15}`) and compare generation
  tok/s across all five values; verify the best-performing configuration
  with a second identical run before recording it as a new best.

## 13. Future Muse optimization experiments

See `experiments/muse-glimmer-30b/optimize.md` for the controlled-optimization
methodology and the running log of what's been tried vs. what's still a
candidate (multi-GPU split strategy, KV cache config, etc.). Draft length
above 15 is **not** a candidate — externally verified `block_size = 16`
confirms 15 is already the drafter's maximum valid value (see §12).
Nothing remaining in that list is assumed to help until benchmarked.
