# Quantization Formats Used In This Repo

This is a technical reference for the formats actually used across the
experiments in this repo. Tradeoffs described are general engineering
characteristics of each format, not claims about universal benchmark
results — always check `docs/experiments.md` / each experiment's
`results.jsonl` for what was actually measured.

## FP16 (float16)

16-bit IEEE floating point. 5 exponent bits, 10 mantissa bits — narrow
dynamic range compared to BF16. Fast on most GPUs with tensor cores, but
can overflow/underflow (→ Inf/NaN) more easily during training-scale
activation ranges, which is exactly what was observed with Gemma 3 4B on
T4 in this repo (`docs/numerical-stability.md`). **Do not assume FP16 is
safe for a new model without validating logits.**

## BF16 (bfloat16)

16-bit float with the same 8-bit exponent range as FP32 but only 7 mantissa
bits — trades precision for dynamic range. This is why BF16 is numerically
stable where FP16 produced NaNs for Gemma 3 4B in this repo: BF16's wider
exponent range avoids the overflow that FP16 hit. Requires Ampere+ GPUs for
native tensor-core acceleration in general, but T4 (Turing) still runs it
correctly, just without the fastest BF16 tensor-core path — it was still
the valid, ~10.04 tok/s baseline observed here.

## NF4 (4-bit NormalFloat, via bitsandbytes)

A 4-bit quantization scheme designed around the assumption that weights are
roughly normally distributed. Weights are dequantized to a compute dtype
(BF16 in this repo) on the fly during the forward pass. Massively reduces
static VRAM for weight storage (~3.2 GB vs. ~8.0 GB BF16 for Gemma 3 4B
here) at the cost of the dequantization overhead per forward pass, which is
consistent with the ~22% lower throughput observed (~7.84 vs. ~10.04 tok/s).

## GPTQ

A post-training quantization method (typically 4-bit) that calibrates
weights layer-by-layer to minimize output error, producing a static
quantized checkpoint (no online dequant-from-full-precision the way NF4
does at load time — GPTQ ships pre-quantized). Fast inference typically
depends on a fused kernel:

- **Marlin** — a highly optimized GPTQ kernel, but it targets newer GPU
  architectures. In this repo it hit a JIT/backend compatibility issue on
  Tesla T4 (compute capability 7.5).
- **`backend="torch"`** — a plain PyTorch dequant fallback, compatible
  with T4 but much slower (~1.8 tok/s observed for Qwen3-30B-A3B on 2×T4
  with a manual layer split).

## GGUF

A single-file container format (successor to GGML) used by llama.cpp. It
bundles the quantized weights, tokenizer, and architecture metadata
together, with an internal versioned quantization scheme (`Q4_0`, `Q4_K_M`,
etc. below). It is llama.cpp/llama-cpp-python's native format and is what
both the Gemma 4 E4B and Muse Glimmer 30B experiments use.

### Q4_0

A simple 4-bit block quantization: each block of weights shares one scale
factor, values quantized to 4-bit integers around it. Cheaper to compute
than the K-quants but generally lower quality per bit than `Q4_K_M` at the
same nominal bit width. Used for the Gemma 4 E4B GGUF in this repo (no
completed benchmark exists for it yet — see `docs/experiments.md` §7).

### Q4_K_M

A "K-quant" scheme: different tensors within the model use different
sub-quantization strategies (a mix of q4_K/q5_K/q6_K/f32 tensors, as
observed in the Muse Glimmer GGUF metadata) chosen to put more precision
where it matters most (e.g. attention/output projections) and less where
it matters least. "M" = medium — a middle point in llama.cpp's Q4_K
size/quality tiers. Observed for Muse Glimmer 30B: ~15.59 GiB file size,
~4.81 bits per weight average.

**Do not confuse a GGUF file's size on disk with VRAM usage** — with
`-ngl 99` the whole quantized model is offloaded into GPU memory, but the
file-size number alone doesn't account for the KV cache, activation
buffers, or CUDA runtime overhead, none of which were separately measured
in this repo.

## Summary table

| Format | Bits | Where used here | Observed here |
|---|---|---|---|
| FP16 | 16 | Gemma 3 4B | **INVALID** (NaN logits) |
| BF16 | 16 | Gemma 3 4B | ~10.04 tok/s, ~8.03 GB VRAM |
| NF4 | 4 | Gemma 3 4B | ~7.84 tok/s, ~3.2 GB VRAM |
| GPTQ Int4 (backend=torch) | 4 | Qwen3-30B-A3B | ~1.8 tok/s, 2× T4 |
| GGUF Q4_0 | 4 | Gemma 4 E4B | Not yet benchmarked |
| GGUF Q4_K_M | ~4.81 avg | Muse Glimmer 30B | ~13.7–17.9 tok/s (config-dependent), 2× T4 |

Avoid drawing unsupported quality conclusions from this table — it records
*what was measured in this repo's specific experiments*, not a general
ranking of these formats.
