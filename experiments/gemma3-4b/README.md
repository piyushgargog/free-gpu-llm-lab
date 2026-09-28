# Experiment 2 — Gemma 3 4B (google/gemma-3-4b-it)

## Environment

- Google Colab Free
- NVIDIA Tesla T4 (~15 GB VRAM)
- CUDA 12.8
- PyTorch 2.11.0+cu128
- Transformers 5.16.1 (at time of experiment)
- Hugging Face authentication required (gated model) — set `HF_TOKEN` in
  a `.env` in the remote environment. **Never commit tokens.**

## Model

`google/gemma-3-4b-it`

## Results summary

| Config | VRAM | Generation speed | Status |
|---|---|---|---|
| FP16 | ~8.6 GB | *(not reportable)* | **INVALID — NaN logits** |
| BF16 | ~8.03 GB | ~10.04 tok/s | Valid observed baseline |
| NF4 (bitsandbytes) | ~3.2 GB | ~7.84 tok/s | Valid observed |

### FP16 — INVALID (numerical-stability failure)

FP16 (`dtype=torch.float16`) initially appeared to load and generate
(~8.6 GB VRAM). Numerical validation of the logits afterward showed:

- **NaN logits**
- Invalid / mostly padding token IDs in the output

**The FP16 benchmark is invalid.** Its apparent generation speed is not
reported anywhere in this repo as a valid number, because a NaN-producing
run isn't actually doing usable inference — see
`docs/numerical-stability.md`.

### BF16 — valid baseline

```python
dtype=torch.bfloat16
attn_implementation="sdpa"
```

- VRAM: ~8.03 GB
- Validation: 0 NaNs, 0 Infs
- Benchmark: 200 generated tokens in ~19.92 s → **~10.04 tokens/sec**

This is the valid Gemma 3 4B baseline used for comparison against NF4.

### NF4 — 4-bit (bitsandbytes)

- VRAM: ~3.2 GB
- Benchmark: 200 generated tokens in ~25.52 s → **~7.84 tokens/sec**

NF4 trades ~60% less VRAM for ~22% less throughput vs. BF16, in this
specific experiment on this specific hardware. This is not a universal
result — see `docs/quantization.md`.

### Attention backend

SDPA (`attn_implementation="sdpa"`) was used successfully. Certain
memory-efficient attention paths failed to engage because Gemma 3 4B uses
grouped-query attention with **8 query heads and 4 KV heads** — some
attention kernel paths assume equal Q/KV head counts. This is documented as
an attention-kernel/backend compatibility issue; unsupported kernels were
never claimed to work.

## Files

- `run.py` — load + generate with selectable dtype (`fp16` / `bf16` / `nf4`). Remote GPU only.
- `validate_logits.py` — forward pass + NaN/Inf logit validation. Remote GPU only.
- `benchmark.py` — timed generation benchmark, records to `results.jsonl` via `free_gpu_llm.benchmarking`. Remote GPU only.
- `gradio_app.py` — local (Colab-hosted) conversational UI, no web retrieval. See below.
- `results.jsonl` — the three historical results above, recorded in the shared benchmark format.

## Gemma 3 Gradio chatbot

A Gradio chat UI (`gradio_app.py`) was built for the Colab environment:

- Conversation history maintained via `free_gpu_llm.generation.build_messages`
- Fixed system prompt (see below)
- Model loaded **once** at process start, not per message
- No secrets hardcoded — `HF_TOKEN` read from environment

System prompt used:

> You are Gemma 3 4B, an AI assistant running locally on a Google Colab
> NVIDIA T4 GPU. You are not Gemini. Answer accurately and concisely. If
> you don't know something, say so rather than inventing information.

This chatbot has no web access — see `experiments/gemma4-e4b` /
`app/gradio_app.py` and `docs/web-retrieval.md` for the later
web-augmented version.

## Reproduction (remote only)

```bash
# In Colab, with HF_TOKEN set:
python experiments/gemma3-4b/run.py --dtype bf16 --prompt "..."
python experiments/gemma3-4b/validate_logits.py --dtype fp16   # reproduces the NaN failure
python experiments/gemma3-4b/benchmark.py --dtype bf16 --max-new-tokens 200
python experiments/gemma3-4b/gradio_app.py
```
