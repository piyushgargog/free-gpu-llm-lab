# Colab / Kaggle Guide

## LOCAL PC vs. REMOTE GPU — read this first

| | LOCAL PC | REMOTE GPU (Colab/Kaggle) |
|---|---|---|
| Coding, docs, git | ✅ | — |
| Lightweight tests (`pytest tests/`) | ✅ | — |
| Static checks (`compileall`) | ✅ | — |
| Download model weights | ❌ **never** | ✅ |
| Load model weights | ❌ **never** | ✅ |
| Run inference | ❌ **never** | ✅ |
| Benchmark a model | ❌ **never** | ✅ |
| Build llama.cpp with CUDA | ❌ (no GPU anyway) | ✅ |

Every script that touches model weights has a `*** REMOTE GPU ONLY ***`
banner at the top of its file. If you're not sure whether a command
downloads/loads a model, don't run it locally — check the script's banner
or ask first.

This guide does **not** include keep-alive hacks for Colab/Kaggle idle
timeouts — none were used in these experiments.

## 1. Start a free GPU runtime

- **Colab:** Runtime → Change runtime type → T4 GPU.
- **Kaggle:** Notebook settings → Accelerator → GPU T4 x2 (Kaggle offers
  2× T4 on its free tier, which is what the Qwen3 and Muse Glimmer
  experiments use).

## 2. Check the GPU

```bash
nvidia-smi
python scripts/check_gpu.py
python scripts/check_environment.py
```

These three commands only *query* hardware/package state — safe to run
first, before installing anything heavier.

## 3. Hugging Face authentication (for gated models, e.g. Gemma)

```bash
cp .env.example .env
# edit .env, set HF_TOKEN=hf_...   (never commit this file)
```

`free_gpu_llm.config.get_hf_token()` reads `HF_TOKEN` from the environment
(loading `.env` via `python-dotenv` if present).

## 4. Install dependencies (remote only)

```bash
pip install -e ".[gpu]"      # transformers-based experiments (Qwen3, Gemma 3)
pip install -e ".[web]"      # Gradio + DDGS + trafilatura
pip install -e ".[llama-cpp]"  # CPU-only llama-cpp-python (see step 6 for CUDA)
```

## 5. Download a model (remote only, guarded)

```bash
python scripts/download_model.py \
  --repo-id google/gemma-3-4b-it \
  --local-dir /content/models/gemma3-4b \
  --confirm-remote-gpu
```

The `--confirm-remote-gpu` flag is required — the script refuses to run
without it, as a guard against accidental invocation.

## 6. Model loading / inference (transformers-based)

```bash
python experiments/gemma3-4b/run.py --dtype bf16 --prompt "..."
python experiments/gemma3-4b/validate_logits.py --dtype bf16
python experiments/gemma3-4b/benchmark.py --dtype bf16 --max-new-tokens 200
```

## 7. Gradio (remote-hosted UI, e.g. via Colab's public URL or Kaggle's port forwarding)

```bash
python experiments/gemma3-4b/gradio_app.py   # no web retrieval
python app/gradio_app.py                     # web-augmented (DDGS + trafilatura)
```

## 8. GGUF models + llama.cpp CUDA build

```bash
git clone https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp
cmake -S /kaggle/working/llama.cpp -B /kaggle/working/llama.cpp/build \
  -DGGML_CUDA=ON -DGGML_CUDA_NO_VMM=ON -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build /kaggle/working/llama.cpp/build -j2

/kaggle/working/llama.cpp/build/bin/llama-cli --list-devices
```

Download a GGUF (remote only, guarded):

```bash
python experiments/gemma4-e4b/download.py \
  --local-dir /kaggle/working/models/gemma4-e4b \
  --confirm-remote-gpu
```

Run/benchmark:

```bash
python experiments/muse-glimmer-30b/run.py \
  --binary /kaggle/working/llama.cpp/build/bin/llama-cli \
  --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf

python experiments/muse-glimmer-30b/benchmark.py \
  --binary /kaggle/working/llama.cpp/build/bin/llama-cli \
  --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \
  --notes "baseline"
```

## 9. DFlash speculative decoding

```bash
python experiments/muse-glimmer-30b/dflash.py \
  --binary /kaggle/working/llama.cpp/build/bin/llama-cli \
  --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \
  --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
```

## 10. Troubleshooting

See `docs/troubleshooting.md` for GPTQ/Marlin, FP16 NaN, attention-kernel,
CPU-only llama-cpp-python, CMake build, and DFlash-warning issues actually
hit during these experiments.
