# Muse Glimmer 30B — llama.cpp native CUDA on 2× Tesla T4

**This is the latest completed experiment and currently the best-optimized
free-GPU setup in this repo.**

## 1. Model

- Source: Meta official Muse Glimmer GGUF repository
- File: `Muse-Glimmer-30B-Q4_K_M.gguf`
- Remote path (Kaggle): `/kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf`
- DFlash drafter: `/kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf`

**These files exist only in the remote Kaggle environment. They are not,
and must never be, downloaded to a personal PC or committed to git** — see
`.gitignore` (`*.gguf`, `models/`).

### Observed model metadata (`llama-cli` output)

| Field | Value |
|---|---|
| Architecture | `muse-glimmer` |
| Model size label | ~28B |
| Block count | 52 |
| Context length | 131072 |
| Embedding length | 6656 |
| Feed-forward length | 19968 |
| Attention heads | 32 |
| KV heads | 2 |
| Sliding window | 2048 |
| Quantization | Q4_K - Medium |
| File size | ~15.59 GiB |
| Bits per weight | ~4.81 BPW |
| Tensor types present | q4_K, q6_K, f32, q5_K |

**~15.59 GiB is the model *file size on disk*, not VRAM usage.** VRAM usage
for this model/config was not separately measured in this experiment — do
not conflate the two.

### DFlash drafter configuration — EXTERNALLY VERIFIED

**This is externally verified architecture/configuration information
about the DFlash drafter, not a result we measured or benchmarked
ourselves.** It was confirmed outside this repo's own artifacts (our own
forensic search of existing logs/docs could not recover it — see
`optimize.md`).

| Field | Value | Source |
|---|---|---|
| `dflash.block_size` | **16** | Externally verified |
| Maximum valid `--spec-draft-n-max` | **15** (= `block_size - 1`) | Derived from the verified `block_size`, per llama.cpp's clamping logic in `common/speculative.cpp` |

This tells us `--spec-draft-n-max 15` (used in the DFlash benchmark below)
was already requesting the **maximum value the implementation accepts** —
not an arbitrary first guess that happened to work. **This is a fact about
the valid configuration range, not a performance claim.** We do not know
from this alone whether 15 is the *fastest* value in that range — only
that it's the *largest* one. See §9b.

## 2. Hardware

2× NVIDIA Tesla T4, ~15–16 GB VRAM each (Kaggle).

```
$ llama-cli --list-devices
CUDA0 Tesla T4
CUDA1 Tesla T4
```

## 3. llama.cpp version / commit

```
0.5.0-dev
build 1
commit 81ef10e
```

## 4. Build configuration

Built from source in Kaggle at `/kaggle/working/llama.cpp`. The default
CMake configuration initially failed around `CUDA::cuda_driver` target
availability; the configuration below is the **observed working build**:

```bash
cmake -S /kaggle/working/llama.cpp \
  -B /kaggle/working/llama.cpp/build \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_NO_VMM=ON \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75

cmake --build /kaggle/working/llama.cpp/build -j2
```

Binary: `/kaggle/working/llama.cpp/build/bin/llama-cli`

Build/runtime configuration included CUDA, NCCL, `USE_GRAPHS=1`,
`NO_VMM=1`, CUDA architecture 750, and Flash Attention support compiled in.

## 5. Baseline command

```bash
MODEL="/kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf"

llama-cli \
  -m "$MODEL" \
  -ngl 99 \
  -c 2048 \
  -n 256 \
  -p "Explain what a transformer neural network is in simple technical terms. Give a concise technical explanation." \
  --no-display-prompt
```

## 6. Baseline benchmark

- Prompt evaluation: **~104.7 tok/s**
- **Generation: ~13.7 tok/s** ← baseline decode benchmark

## 7. Batch size experiment (`-b 512`)

- Prompt: ~110.4 tok/s
- Generation: ~13.6 tok/s
- **Conclusion: no useful decode-speed improvement.**

## 8. Flash Attention experiment (`-fa on`)

- Prompt: ~109.7 tok/s
- Generation: ~13.7 tok/s
- **Conclusion: no measurable decode-speed improvement in this configuration.**

## 9. DFlash speculative decoding experiment

Currently the most important successful Muse optimization.

```bash
MODEL="/kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf"
DRAFT="/kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf"

llama-cli \
  -m "$MODEL" \
  -md "$DRAFT" \
  --spec-type draft-dflash \
  --spec-draft-n-max 15 \
  -ngl 99 \
  -c 2048 \
  -n 256 \
  -p "Explain what a transformer neural network is in simple technical terms. Give a concise technical explanation." \
  --no-display-prompt
```

- **Generation: ~17.9 tok/s** — this is our own measured benchmark result.
- Baseline: ~13.7 tok/s — our own measured benchmark result.
- Improvement: ~4.2 tok/s → **~30.7% higher generation throughput (~1.31× baseline)**
- Prompt tok/s for this configuration: **not measured (TBD)** — do not invent it.
- `--spec-draft-n-max 15` is, separately, now known to be the **maximum
  valid value** for this drafter (externally verified `block_size = 16`,
  see the DFlash drafter configuration table above). That's a fact about
  the configuration's valid range, not a claim that 15 is the *fastest*
  value — performance across the valid range is still unmeasured (§9b).

## 9b. DFlash draft-length sweep (`--spec-draft-n-max`) — NOT RUN

`--spec-draft-n-max 15` was the first value tried and happens to be the
**maximum valid value** — externally verified: the drafter's
`dflash.block_size = 16`, and llama.cpp's `common/speculative.cpp` clamps
`--spec-draft-n-max` to `block_size - 1 = 15` for standard (non-anchor)
DFlash. This means the sweep below (`8` through `15`) **covers the entire
valid configuration range** — there is no higher value to test without
changing the drafter itself. Also per llama.cpp's `docs/speculative.md`,
**`llama-cli` does not print draft acceptance statistics** (only
`llama-server` does), so this sweep's metrics are generation tok/s
(primary) and prompt tok/s (secondary, when parseable) only — no
accepted/drafted/acceptance-ratio numbers.

**We do not yet know which value in `{8, 10, 12, 14, 15}` is fastest.**
Knowing that 15 is the maximum *valid* value says nothing about whether
it is the maximum *performing* value — that requires the benchmark below
to actually run.

Harness: `sweep_draft_length.py`. Exact remote command:

```bash
python experiments/muse-glimmer-30b/sweep_draft_length.py \
  --binary /kaggle/working/llama.cpp/build/bin/llama-cli \
  --model /kaggle/working/models/muse-glimmer/Muse-Glimmer-30B-Q4_K_M.gguf \
  --draft /kaggle/working/models/muse-glimmer/dflash-Muse-Glimmer-30B-Q4_K_M.gguf
```

| `--spec-draft-n-max` | Generation tok/s | Status |
|---:|---:|---|
| 8 | — | NOT RUN |
| 10 | — | NOT RUN |
| 12 | — | NOT RUN |
| 14 | — | NOT RUN |
| 15 (re-verification) | — | NOT RUN |

**Status: NOT RUN.** No remote (Kaggle) execution is available from this
dev environment. **17.9 tok/s (`--spec-draft-n-max 15`) remains the
current best until one of these is actually benchmarked and beats it.**
Full methodology and rationale: `optimize.md`.

### Next experiment

Once Kaggle is available, run the prepared draft-length sweep
(`--spec-draft-n-max` ∈ `{8, 10, 12, 14, 15}`, command above) and compare
generation tok/s across all five values. If any value beats the current
17.9 tok/s best, verify it with a second identical run
(`sweep_draft_length.py` does this automatically) before recording it as
the new best. This sweep now covers the drafter's entire valid
`--spec-draft-n-max` range (externally verified `block_size = 16` — see
the DFlash drafter configuration table above), so no further sweep
extension is needed unless the drafter itself changes.

## 10. Current best result

**DFlash speculative decoding, ~17.9 tok/s generation.** This is the
current best-observed Muse configuration and is the number all future
optimization attempts must be compared against — **as long as DFlash
remains enabled.** Do not compare a future optimized-but-DFlash-disabled
run against 17.9 tok/s and call it a regression, and do not compare a
future DFlash-enabled run against the old 13.7 tok/s no-DFlash baseline
and call it an improvement.

## Current benchmark table

| Configuration | Prompt tok/s | Generation tok/s | Status |
|---|---:|---:|---|
| Muse baseline | ~104.7 | ~13.7 | Valid observed |
| Muse + batch 512 | ~110.4 | ~13.6 | No decode improvement |
| Muse + Flash Attention | ~109.7 | ~13.7 | No decode improvement |
| Muse + DFlash | TBD | ~17.9 | **Best observed** |
| Muse + DFlash, `spec-draft-n-max` sweep {8,10,12,14,15} | — | NOT RUN | Harness ready, see §9b |

Machine-readable version: `results.jsonl` (append-only; see
`free_gpu_llm.benchmarking`).

## 11. Limitations

- Prompt-eval tok/s for the DFlash configuration was never measured.
- No multi-GPU tensor/layer-split, NCCL, CUDA-graph, or KV-cache-config
  experiments have been run yet (see `optimize.md`).
- Only a single fixed prompt/length (256 generated tokens, `-c 2048`) has
  been benchmarked; results may not generalize to longer contexts or
  different prompts.
- All numbers are single-run observations, not averaged over multiple runs.

## 12. Reproducibility instructions

All of the above requires a remote GPU environment (Kaggle, 2× T4). See
`docs/colab.md`. Summary:

1. Build llama.cpp with the CMake configuration in §4.
2. Obtain `Muse-Glimmer-30B-Q4_K_M.gguf` and the DFlash drafter GGUF (not
   included in this repo — remote only).
3. Run the commands in §5–§9.
4. Record results with `scripts/benchmark.py --log-file <captured output> ...`
   or `experiments/muse-glimmer-30b/benchmark.py`.

## 13. Known warnings

During DFlash startup, llama.cpp prints:

```
dflash requires ctx_other to be set
(this warning is normal during memory fitting)
```

The program continues and successfully generates output. **This is an
initialization/memory-fitting warning, not a fatal failure** — it is not
hidden from users of this repo, and it does not mean DFlash failed.

## 14. Future optimization work

Per the controlled-optimization methodology (`optimize.md`), candidate
areas to investigate remotely — none of these are assumed to help; each
must be benchmarked against the 17.9 tok/s DFlash baseline before being
kept:

- llama.cpp CUDA kernels / CUDA graphs
- Multi-GPU tensor/layer split strategy, inter-GPU communication, NCCL behavior
- KV cache configuration, context size, batch configuration
- DFlash draft length (`--spec-draft-n-max`) — **harness ready, NOT RUN yet, see §9b**
- Speculative acceptance rate (requires `llama-server`, not `llama-cli` — see §9b)
- CUDA architecture-specific behavior, T4-specific kernel paths
- Memory transfer overhead, CPU/GPU synchronization
- Quantized kernel selection

## Files

- `benchmark.py` — CLI to record a Muse benchmark run (parses captured `llama-cli` output).
- `run.py` — thin wrapper to invoke `llama-cli` with a given config (remote only; does not itself download/load anything on your behalf beyond spawning the `llama-cli` binary).
- `dflash.py` — DFlash-specific run wrapper (baseline vs. DFlash comparison).
- `sweep_draft_length.py` — controlled `--spec-draft-n-max` sweep (§9b). Remote only; not yet run.
- `optimize.md` — controlled optimization methodology + log of what's been tried.
- `results.jsonl` — the four historical results above, in the shared benchmark format. The draft-length sweep will append further rows here when actually run.
