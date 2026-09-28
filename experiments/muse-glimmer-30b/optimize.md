# Muse Glimmer 30B — Optimization Methodology & Log

This is a controlled optimization process, not random flag testing.

## Methodology

1. Establish baseline.
2. Change **one** variable.
3. Rebuild only if required.
4. Run the **same** benchmark (same prompt, same `-n 256`, same `-c 2048`).
5. Record generation tok/s (`benchmark.py`, which appends to `results.jsonl`).
6. Compare against the current best.
7. Keep only demonstrated improvements.
8. Revert failed/neutral experiments.
9. Document *why* an optimization worked or failed, not just the number.

## Current state

- **Baseline:** ~13.7 tok/s (no speculative decoding) — our own measured benchmark result.
- **Current best:** ~17.9 tok/s (DFlash speculative decoding, `--spec-draft-n-max 15`) — our own measured benchmark result.
- **DFlash drafter `block_size` = 16 — EXTERNALLY VERIFIED** (not recovered from our own logs/artifacts — our own forensic search of this repo could not find it; see "DFlash draft-length sweep" below). This makes **`--spec-draft-n-max 15` the maximum value the implementation accepts** for standard (non-anchor) DFlash (`block_size - 1 = 15`, per `common/speculative.cpp`'s clamping logic). This is a fact about the *valid configuration range*, not a performance finding — it does **not** mean 15 is the fastest value in that range, only the largest.

Every future experiment must state which of the two *measured* numbers
above it is being compared against, and must keep DFlash enabled when
comparing against the 17.9 tok/s number — a DFlash-disabled run compared
against 17.9 tok/s is a regression by construction, not a real finding.

## Log

| # | Change | Compared against | Result | Kept? | Why |
|---|---|---|---|---|---|
| 1 | Baseline (`-ngl 99 -c 2048 -n 256`) | — | 13.7 tok/s gen, 104.7 tok/s prompt | — | Establishes baseline |
| 2 | `-b 512` (batch size) | Baseline (13.7) | 13.6 tok/s gen, 110.4 tok/s prompt | No | Batch size affects prompt-processing parallelism, not single-stream autoregressive decode throughput, which is why prompt tok/s rose (~110.4) while generation tok/s did not (~13.6) |
| 3 | `-fa on` (Flash Attention) | Baseline (13.7) | 13.7 tok/s gen, 109.7 tok/s prompt | Neutral, not adopted as a "win" | Decode speed at this context length (2048) is likely dominated by weight-loading/dequant bandwidth rather than attention compute, so a faster attention kernel doesn't move the needle here |
| 4 | DFlash speculative decoding (`--spec-draft-n-max 15`) | Baseline (13.7) | **17.9 tok/s gen** | **Yes — new best** | Speculative decoding lets the small drafter propose multiple tokens verified in one batched forward pass of the big model, amortizing the big model's per-token overhead — a fundamentally different mechanism from attention-kernel or batch-size tuning, which is why it's the first change that moved decode throughput |
| 5 | DFlash draft-length sweep (`--spec-draft-n-max` ∈ {8,10,12,14,15}) | DFlash best (17.9) | **NOT RUN** | Pending | Harness ready (`sweep_draft_length.py`); no remote (Kaggle) execution is available from this dev environment. See "DFlash draft-length sweep" section below. |

## DFlash draft-length sweep (spec-draft-n-max) — NOT RUN

**Hypothesis:** `--spec-draft-n-max 15` was carried over from the first
working DFlash command and was never itself swept — a different draft
length might trade off draft-generation cost against acceptance rate
differently and yield higher decode tok/s. **We only know 15 is the
maximum valid value (see below) — we do NOT know it is the fastest one.**

**Externally verified configuration fact (not our own measurement, not
recovered from our own logs — see forensic note below):**

- **`dflash.block_size = 16`** for Muse Glimmer's DFlash drafter.
- Per `common/speculative.cpp`'s clamping logic, `--spec-draft-n-max` is
  clamped to `block_size - 1 = 15` for standard (non-anchor) DFlash.
  **15 is therefore the maximum valid value** the implementation accepts
  for this drafter — not an arbitrary first guess that happened to work.
- **Forensic note:** a dedicated search of every artifact already in this
  repo (docs, READMEs, results, code) turned up **no** captured
  `llama-cli` console output containing this value — the `block_size=16`
  fact came from outside this repo's own artifacts, not from something we
  recovered here. See `experiments/muse-glimmer-30b/optimization-research.md`
  for the source-code-level explanation of *why* `--spec-draft-n-max` is
  clamped this way (the clamping mechanism itself was found via primary
  llama.cpp source inspection; the specific `block_size=16` value for
  *this* drafter was confirmed externally).
- Per `llama.cpp`'s `docs/speculative.md` (and confirmed via call-site
  search of `common_speculative_print_stats`): `llama-cli` (what this
  repo's benchmark harness uses, for consistency with every other Muse
  result) **does not print draft acceptance statistics** — only
  `llama-server`'s per-slot output does. So this sweep's secondary
  metrics are limited to prompt tok/s (when parseable);
  accepted/drafted/acceptance-ratio numbers are **not available** without
  switching to `llama-server`, which would be a different variable and is
  out of scope for this experiment.

**Practical consequence:** the sweep `{8, 10, 12, 14, 15}` now covers the
**entire valid `--spec-draft-n-max` range** for this drafter — there is
no higher value to test without a different drafter file. Nothing above
15 needs to be attempted; `sweep_draft_length.py`'s refusal of values > 15
without `--allow-above-15` is therefore correct as-is and does not need
to change.

**Configuration (identical to the DFlash baseline except `--spec-draft-n-max`):**

- Model: `Muse-Glimmer-30B-Q4_K_M.gguf`, draft: `dflash-Muse-Glimmer-30B-Q4_K_M.gguf`
- `-ngl 99 -c 2048 -n 256`, same prompt as every other Muse benchmark in this repo
- No `-b`, no `-fa` (both already shown not to help; not stacked here to keep one variable at a time)
- `--spec-type draft-dflash`, `--spec-draft-n-max` swept over `{8, 10, 12, 14, 15}`

**Harness:** `experiments/muse-glimmer-30b/sweep_draft_length.py`. Runs
each value once, appends every run (including failures) to
`results.jsonl`, and if a candidate beats 17.9 tok/s, automatically
re-runs it once more before it's allowed to be reported as a new best.
Refuses any value > 15 unless `--allow-above-15` is passed.

**Exact Kaggle command:**

```bash
python experiments/muse-glimmer-30b/sweep_draft_length.py \
  --binary /kaggle/working/llama.cpp/build/bin/llama-cli \
  --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \
  --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
```

**Status: NOT RUN.** This dev environment has no remote execution access
to Kaggle. Run the command above there; results append to `results.jsonl`
automatically. Report back the console table (or the appended
`results.jsonl` lines) and this doc / `README.md` will be updated with
the real numbers.

### The real ceiling — RESOLVED

Previously this section asked "confirm the drafter's trained block size
before ever passing `--allow-above-15`." **That is now answered:
`dflash.block_size = 16`, so the maximum valid `--spec-draft-n-max` is
15 (externally verified — see above).** There is no remaining open
question about the ceiling itself. A `gguf_dump.py` metadata dump (as
previously suggested here) would be redundant for confirming the ceiling,
though it remains a reasonable way to independently cross-check the
value if ever desired:

```bash
python /kaggle/working/llama.cpp/gguf-py/scripts/gguf_dump.py \
  /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
```

No `--allow-above-15` sweep is planned or needed — 15 is the ceiling, not
an unconfirmed guess.

### Caution: "ctx_other" warning during DFlash startup

The historical baseline confirmed this message is non-fatal in our setup
(generation completed normally — see README.md §13). However, external
llama.cpp fork issue trackers describe the same/similar message
(`<arch> requires ctx_other to be set`) preceding a **hard failure** (null
draft context, crash) in certain speculative-decoding code paths on other
forks/configurations. If any sweep run at a new `--spec-draft-n-max` value
crashes or segfaults after printing this message, treat that specific
value as **FAILED** (per the experiment protocol) rather than assuming
it's the same harmless warning — don't extrapolate the "harmless" finding
from n_max=15 to every other value without checking.

## Not yet tried (candidates, un-benchmarked)

Per the historical instructions for this repo, none of these are assumed
to help — each needs its own controlled run against the 17.9 tok/s DFlash
baseline before being written up as a result:

- Multi-GPU tensor/layer split strategy (vs. the default llama.cpp split)
- Inter-GPU communication / NCCL tuning
- KV cache configuration and quantization
- Different context sizes (currently only `-c 2048` tested)
- ~~DFlash draft length above 15~~ — **not applicable**: 15 is the drafter's externally verified maximum valid value (`block_size = 16`); there is no higher value to test without a different drafter file
- Speculative acceptance rate analysis (requires `llama-server`, not `llama-cli`)
- CUDA architecture-specific kernel paths beyond `sm_75`
- Memory transfer overhead / CPU-GPU synchronization profiling
- Alternative quantized kernel selection

## Known non-fatal warning

```
dflash requires ctx_other to be set
(this warning is normal during memory fitting)
```

Printed during DFlash startup; generation completes successfully
afterward. Documented here (and in README.md) rather than hidden.
