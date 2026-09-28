# Numerical Stability

## Why this matters

A model can appear to "work" — load without error, generate text of the
right shape/length — while being numerically broken. The only reliable way
to catch this is to inspect the actual logit values, not just whether
`generate()` returned something.

## Case study: Gemma 3 4B FP16 on Tesla T4

`google/gemma-3-4b-it` loaded with `dtype=torch.float16` appeared to work:
it loaded (~8.6 GB VRAM) and produced output of the expected shape.
Validating the logits afterward showed:

- **NaN logits present**
- Output token IDs were invalid / mostly padding tokens

**Conclusion: the FP16 benchmark for this model is INVALID.** Its apparent
generation speed is never reported as a real number anywhere in this repo
— a NaN-producing forward pass isn't doing usable inference, so timing it
measures nothing meaningful.

Switching to `dtype=torch.bfloat16` (same `attn_implementation="sdpa"`,
same hardware) validated clean — 0 NaNs, 0 Infs — and became the valid
baseline (~10.04 tok/s). BF16's wider exponent range (same as FP32, unlike
FP16's narrower range) is the likely reason it avoids the overflow that
produced NaNs under FP16; this wasn't independently root-caused further
than that in the original experiment.

## How validation is done in this repo

`free_gpu_llm.generation.validate_logits()` (`src/free_gpu_llm/generation.py`)
checks a flat iterable of logit values for NaN/Inf and returns a
`LogitValidation(total, nan_count, inf_count)` whose `.valid` property is
`False` if either count is nonzero. It accepts torch tensors directly (via
`.flatten().tolist()`) or plain iterables of floats, so it's usable both
in a real forward pass (remote GPU) and in unit tests (local, no GPU —
see `tests/test_history.py`).

`experiments/gemma3-4b/validate_logits.py` and `benchmark.py` both run this
check before trusting a timing result: `benchmark.py` specifically forces
`tokens_per_second=0.0` and `valid=False` in the recorded `BenchmarkResult`
if validation fails, rather than recording a timing number that doesn't
mean anything.

## Rule for this repo

**Any run with NaN or Inf logits is invalid and must be recorded/reported
as such (`valid: false` in `BenchmarkResult`/`results.jsonl`), regardless
of whether it "looked like it worked."** Do not report its speed. Do not
imply it's a usable configuration.
