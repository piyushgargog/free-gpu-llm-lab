# Muse Glimmer Optimization Research

**Status: SOURCE INSPECTION ONLY. No benchmarks were run to produce this
document — Kaggle is currently unavailable.** Every claim below is either
(a) sourced directly from the `ggml-org/llama.cpp` mainline repository as
of 2026-09-28 (file/line cited), or (b) explicitly flagged as an
unconfirmed hypothesis. Nothing here should be read as a performance
result.

## Current Verified Baseline

Muse-Glimmer-30B-Q4_K_M, 2× Tesla T4, llama.cpp, no speculative decoding:
**13.7 tok/s** generation, ~104.7 tok/s prompt eval. (`-ngl 99 -c 2048 -n 256`)

## Current Best

Same config + DFlash speculative decoding (`--spec-type draft-dflash
--spec-draft-n-max 15`): **17.9 tok/s** generation (~30.7% over baseline).
The draft-length sweep over `{8,10,12,14,15}` is prepared
(`sweep_draft_length.py`) but **NOT RUN**.

## Architecture / Execution Path

Source: `src/models/dflash.cpp`, `common/speculative.cpp` (struct
`common_speculative_impl_draft_dflash`, `ggml-org/llama.cpp`).

DFlash is a **block-diffusion drafter**, not an autoregressive draft model
(unlike EAGLE/MTP-style drafters in the same codebase). Per decode step:

1. **`process()`** — every time the target model runs, the target's hidden
   states at a configured set of `target_layer_ids` layers are read back
   to CPU (`llama_get_embeddings_layer_inp`), copied **token-by-token,
   layer-by-layer via `memcpy`** into a separate `batch_inject` buffer,
   then handed to a dedicated `llama_decode(ctx_dft, batch_inject)` call
   that injects them into the **draft's own, separate KV cache** via
   `cpy_k`/`cpy_v` (`src/models/dflash.cpp:608-676`). This is a real,
   source-confirmed CPU-mediated data path between target and draft,
   distinct from the target's own decode.
2. **`draft()`** — builds a "noise block" `[id_last, <mask> ×
   (block_size-1)]` and runs it through the draft's own transformer layers
   with **non-causal, cache-aware attention** over `[committed, MASK...]`
   positions (`dflash.cpp:569-853`, comment: "token batch -> noise-block
   diffusion: attend over [committed, MASK...] to generate draft tokens").
   This produces up to `block_size-1` draft tokens in one forward pass,
   which is *why* DFlash amortizes cost differently than a classic
   one-token-at-a-time draft model.
3. **Verification** — the target model verifies the drafted block in a
   single batched forward pass (standard speculative-decoding
   verification, `common_speculative_accept` in `common/speculative.cpp`).
4. **`ctx_other` sharing** — if the drafter's GGUF doesn't ship its own
   token-embedding/output-head tensors (a common space-saving choice for
   drafters), `dflash.cpp:679-687` and `:800-808` fall back to
   `cparams.ctx_other` to borrow the **target model's** `tok_embd`/`output`
   tensors directly. This is exactly what the documented `"<arch>
   requires ctx_other to be set"` startup message refers to — it is not a
   mysterious warning, it's the code checking whether that pointer has
   been wired up yet during the memory-fitting pass.

Two DFlash variants exist in the same struct, selected by GGUF metadata,
and **we do not currently know which one Muse Glimmer's drafter is**:

- **DFlash1** (`is_dflash2 = false`): samples directly from the draft's
  own logits each masked position.
- **DFlash2** (`is_dflash2 = (selector_top_k > 0)`, reads
  `selector_hidden.weight` tensor): additionally builds a top-k candidate
  **selector lattice** on GPU (`build_dflash2_selector`,
  `dflash.cpp:478-567`) that is walked on the CPU side per the code
  comment `"packed into the nextn output slot for the CPU-side walk"` —
  an extra CPU-side step not present in DFlash1.
- A third variant, **DSpark**, adds a Markov head
  (`build_dspark_markov_head`) — used for a different family of models,
  not indicated as relevant to Muse Glimmer by anything we've observed,
  but technically the same code path.

**This is resolvable for free**: DFlash's constructor prints
`LOG_INF: "... block_size=%d, mask_token_id=%d, n_extract=%u,
sample_from_anchor=%s"` (`speculative.cpp:993-995`) and
`load_arch_tensors` prints `"DFlash2 conv kernel = ..., selector rank =
..., top-k = ..."` only if DFlash2 tensors are present
(`dflash.cpp:156-158`). **If the original 17.9 tok/s run's full console
output was ever saved, it already answers this — check before running
anything new on Kaggle.**

## CUDA Graph Analysis

Source: `ggml/src/ggml-cuda/ggml-cuda.cu:2554-4493`,
`ggml/src/ggml-cuda/common.cuh:1262-1293`.

- CUDA graph support is compiled in via the `USE_CUDA_GRAPH` macro
  (CMake-controlled, not something we pass explicitly in our build
  command — its default depends on the llama.cpp CMake version; our
  recorded build command never sets `-DGGML_CUDA_GRAPHS=OFF`, and the
  historical build log reports `USE_GRAPHS=1`, so it is compiled in for
  our build).
- **Runtime enable/disable, no rebuild required:** `graph->is_enabled()`
  (`common.cuh:1288-1291`) returns false if either (a) the GPU's compute
  capability is below Volta (`GGML_CUDA_CC_VOLTA = 700`) — **T4 is 750,
  so this does NOT disable graphs on our hardware** — or (b) the
  environment variable **`GGML_CUDA_DISABLE_GRAPHS`** is set to any
  non-empty value. This is the actual, source-confirmed runtime toggle —
  not guessed from a CLI `--help` listing.
- **Warmup/capture mechanism** (`ggml-cuda.cu:4436-4493`): a CUDA graph is
  only replayed (fast path) after **two consecutive calls with identical
  graph node properties** (shapes, source data pointers — checked via
  `ggml_cuda_graph_update_required`, `ggml-cuda.cu:2592-2632`). If
  properties change between calls, `warmup_complete` resets to false and
  execution falls back to direct (non-graph) mode until two stable calls
  occur again. The graph is keyed by `cgraph->nodes[0]`
  (`ggml_cuda_graph_get_key`, `:2588-2590`) — i.e. graphs of different
  *shapes* are tracked as different graph instances, not merged.
- **Hypothesis (unconfirmed, plausible from source):** DFlash's decode
  step alternates between structurally different graphs — the KV
  "injection" graph (`ubatch.embd` branch, `dflash.cpp:609`) and the
  "diffusion" draft graph (`dflash.cpp:689` branch) are different shapes,
  and the diffusion graph's own tensor shapes depend on `n_block_tokens`
  (`speculative.cpp:1202`), which in turn depends on the requested
  `--spec-draft-n-max`. If any of these shapes vary step-to-step (e.g.
  due to varying accepted-token counts changing subsequent batch sizes),
  CUDA graphs may never reach a stable "replay" state for that particular
  graph shape, paying repeated capture overhead instead. **This is a
  hypothesis, not a measured finding** — it would explain a smaller
  DFlash speedup than a naive "block diffusion should amortize cost N×"
  estimate might predict, but we have no profiling data to confirm it.
- A second, separate, **off-by-default** env var,
  `GGML_CUDA_GRAPH_OPT=1` (`ggml-cuda.cu:4621-4628`), enables an
  additional stream-parallelism optimization pass over an already-captured
  graph (fan-out-based multi-stream scheduling of independent branches).
  Its interaction with DFlash's graph shapes is not documented and not
  tested anywhere we found.
- `GGML_CUDA_NO_VMM=ON` (used in our build) is unrelated to CUDA graphs —
  see the T4/SM75 section below.

## Multi-GPU Analysis

Source: `docs/multi-gpu.md` (fetched directly, 2026-09-28), corroborated
by `ggml/cmake/FindNCCL.cmake`, `ggml/src/ggml-cuda/allreduce.cu`.

- **Default split mode is `layer` (pipeline parallel)**: each GPU holds a
  contiguous slice of transformer layers, and the KV cache for layer `l`
  lives on the GPU that owns layer `l`. None of our recorded Muse Glimmer
  commands ever pass `--split-mode`/`-sm` or `--tensor-split`/`-ts`, so
  **every benchmark we have used this default automatic, memory-proportional
  layer split** across the 2× T4s.
- **NCCL is compiled in by default (`-DGGML_CUDA_NCCL=ON`) but is only
  actually *used* for cross-GPU reductions in `tensor` split mode.** Since
  we've never requested tensor/row split, **NCCL was very likely present
  in the binary but not exercised** in any of our benchmarked runs,
  despite the original build notes listing "NCCL" as part of the runtime
  configuration. This is a documentation clarification, not a performance
  finding — flagging it so a future reader doesn't assume NCCL traffic
  was part of our 13.7→17.9 tok/s story.
- **`--split-mode tensor`** (the modern name for what used to be called
  `row` split, now "experimental") splits both weights *and* KV cache
  across GPUs via tensor parallelism, and is explicitly documented as
  optimizing **decode latency** (our exact metric) at the cost of more
  cross-GPU communication per layer — the doc states plainly:
  *"pipeline-parallel maximizes batch throughput; tensor-parallel
  minimizes latency."* This is the single most directly-relevant,
  documented lever for our specific single-stream decode-tok/s workload
  that we have never tried. Caveat: it requires a reasonably fast
  interconnect; Kaggle's paired T4s are commonly PCIe-only (no NVLink) —
  we have not confirmed the actual interconnect topology, so whether the
  added cross-GPU traffic in tensor mode nets positive on Kaggle
  specifically is **genuinely unknown**, not just untested.
- **`--tensor-split N,M`** allows manually overriding the automatic
  per-GPU allocation ratio. With two identical T4s (same VRAM), the
  automatic memory-proportional split should already be even; we have no
  source or metadata evidence of a per-layer compute imbalance in Muse
  Glimmer specifically that would call for manual rebalancing. Worth
  checking only if `nvidia-smi`/`nvtop` during a run shows one GPU
  finishing its layers well before the other.

## DFlash Analysis

(See also Architecture section above for the full process/draft/verify
cycle — this section focuses on the `--spec-draft-n-max` question
specifically.)

Source: `common/speculative.cpp:920-1006` (constructor of
`common_speculative_impl_draft_dflash`).

```cpp
block_size = 16;  // fallback if the GGUF has no dflash.block_size key
if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
    block_size = std::atoi(buf);
}
...
const int32_t n_draft_max = is_dspark && sample_from_anchor ? block_size : block_size - 1;
if (this->params.n_max > n_draft_max || this->params.n_min > n_draft_max) {
    LOG_WRN("... requested draft size (n_max=%d, ...) exceeds the trained block size %d -- clamping to %d\n", ...);
    this->params.n_max = std::min(this->params.n_max, n_draft_max);
}
```

- **`--spec-draft-n-max` is clamped to `block_size - 1`** for standard
  DFlash (non-anchor-sampling), where `block_size` is read from the
  drafter GGUF's `dflash.block_size` key at runtime (the `= 16` above is
  the code's own fallback if that key were absent — see below for the
  actual value).
- **UPDATE (post-dated this section's original research pass):
  Muse Glimmer's drafter `dflash.block_size = 16` is now EXTERNALLY
  VERIFIED** (confirmed outside this repo's own artifacts — a dedicated
  forensic search of every existing file/log/doc in this repo could not
  recover it from anything we had captured ourselves). This is **not** a
  benchmark result and **not** something we derived from the code's
  default fallback value coinciding with the real one by assumption — it
  is a separately confirmed fact about this specific drafter file.
  Consequently: **`n_draft_max = 16 - 1 = 15`, so `--spec-draft-n-max 15`
  is confirmed to be the implementation's maximum valid value** for this
  drafter, not an arbitrary first-guess that happened to work. This is a
  fact about the *valid configuration range* — it says nothing about
  which value in that range performs best. This also confirms:
  - **The existing draft-length sweep (`8,10,12,14,15`) covers the
    entire valid range** — nothing above 15 can ever be exercised (it
    would clamp back down to 15, with a warning, per the code above).
  - The sweep's job is to determine whether a **shorter** draft length
    (trading fewer wasted-drafted-and-rejected tokens against less
    per-block amortization) performs *better* than 15 — not to search
    for a higher ceiling, which doesn't exist for this drafter.
  - **We still do not know which of `{8,10,12,14,15}` is fastest** — that
    remains unbenchmarked and is the actual next experiment.
- **CPU-side overhead**: confirmed in source (see Architecture section)
  — `process()`'s per-token, per-extract-layer `memcpy` loop plus a
  dedicated `llama_decode(ctx_dft, ...)` call happens on every target
  step, not just at block boundaries. Whether this is a measurable
  fraction of the ~13.7→17.9 tok/s window is **not profiled or
  estimated here** — flagged as a real mechanism, not a quantified cost.
- **Graph capture involvement**: yes, per the CUDA Graph Analysis section
  — the draft's decode calls go through the same `ggml_backend_cuda_graph_compute`
  path as any other CUDA-backend decode; whether the draft's graphs
  reach warmup-complete steady state is unconfirmed.
- **Values below/above 15**: below 15 is unrestricted (any `1..15`, per
  the sweep already prepared); above 15 is technically *accepted* by the
  CLI parser but **silently clamped** at runtime with a warning — not a
  hard error, but not actually testing the requested value either.

## CUDA Kernel Analysis

Source: `ggml/src/ggml-cuda/mmq.cuh:230-263`,
`ggml/src/ggml-cuda/common.cuh:52-375`.

```cpp
static ggml_cuda_mmq_config ggml_cuda_mmq_get_config(...) {
    ...
    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_VOLTA) {   // 700
        return ggml_cuda_mmq_get_config_ampere(type, J, fallback);
    }
    if (ggml_cuda_highest_compiled_arch(cc) >= GGML_CUDA_CC_DP4A) {
        return ggml_cuda_mmq_get_config_pascal_dp4a(type, J, fallback);
    }
    return ggml_cuda_mmq_get_config_pascal_older(type, J, fallback);
}
```

- The MMQ (quantized matrix-multiply, used for Q4_K decode) kernel tuning
  tables are per-architecture-family (`mmq-config-pascal-older.cuh`,
  `-pascal-dp4a.cuh`, `-ampere.cuh`, `-blackwell.cuh`, plus AMD variants).
  **There is no Turing-specific table**, despite `GGML_CUDA_CC_TURING =
  750` being a distinctly defined constant elsewhere in the codebase
  (`common.cuh:53`). Any NVIDIA GPU with `cc >= 700` that isn't Blackwell
  gets the **Ampere-tuned** tile size / thread count / occupancy
  parameters. T4 (`cc = 750`) falls into this bucket.
- This is real (source-confirmed selection logic) but its actual
  performance impact on T4 is **unverified** — Ampere and Turing share
  the `dp4a`/int8 tensor-core instruction family closely enough that the
  tables may still be reasonably close to optimal, or may not be; we have
  no profiling evidence either way.
- `TURING_MMA_AVAILABLE` is defined for `__CUDA_ARCH__ >= 750`
  (`common.cuh:288-290`), confirming **T4 does use the MMA (tensor-core)
  quantized-matmul code path**, not a naive/scalar fallback.
- `CP_ASYNC_AVAILABLE` requires `__CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE`
  (800) (`common.cuh:300-302`) — **T4 structurally lacks async-copy
  (`cp.async`)**, which Ampere+ GPUs use to hide global-memory latency
  during kernel data loads. This is a hardware ceiling, not a
  configuration choice — nothing in llama.cpp's config surface can add
  `cp.async` support to Turing.

## KV Cache Analysis

- **Two separate KV caches exist per DFlash run**: the target model's own
  cache (sized by Muse Glimmer's `sliding_window: 2048`, per the
  already-documented GGUF metadata in `README.md`), and the **draft
  model's own, separate KV cache**, into which target hidden-state
  features get injected (`dflash.cpp:608-676`, confirmed in Architecture
  section above). This means total KV VRAM usage is higher than a
  non-speculative run's — not measured/quantified here.
- DFlash's own decode attention is explicitly **non-causal** over
  `[committed, MASK...]` positions (`llama_set_causal_attn(ctx_dft,
  causal_attn)` where `causal_attn` defaults to `false` unless the GGUF's
  `dflash.attention.causal` key says otherwise,
  `speculative.cpp:937,973-975,1050`) — a fundamentally different
  attention pattern from the target's normal causal decode, using a
  dedicated masked-attention graph builder
  (`build_attn_inp_kv`/`build_attn_inp_kv_iswa`,
  `dflash.cpp:584-590`).
- Interleaved sliding-window attention (iSWA) routing code exists in the
  DFlash graph builder (`use_iswa`, `inp_attn_iswa`,
  `dflash.cpp:582-590,643-659`) — **whether Muse Glimmer's target model
  itself uses standard or per-layer-interleaved SWA is not determinable**
  from the single `sliding_window: 2048` value already recorded in
  `README.md`; that field doesn't distinguish the two.

## T4 / SM75 Analysis

Consolidated from the sections above — no new source citations, this is
a summary specific to our hardware:

| Property | T4 (cc 750) | Source |
|---|---|---|
| CUDA graphs | Enabled (not excluded by the `cc < Volta` check) | `common.cuh:424-428` |
| MMQ kernel tuning table | Shares Ampere's table (no Turing-specific one exists) | `mmq.cuh:256-257` |
| Tensor-core MMA path | Available (`TURING_MMA_AVAILABLE` for `cc>=750`) | `common.cuh:288-290` |
| `cp.async` (async global-memory load) | **Not available** (Ampere+ only, `cc>=800`) | `common.cuh:300-302` |
| `GGML_CUDA_NO_VMM=ON` in our build | Disables CUDA's virtual-memory-management pool allocator (`GGML_USE_VMM`), falling back to plain `cudaMalloc` | `common.cuh:231,263-265` |

**On `GGML_CUDA_NO_VMM`:** this was originally adopted in our build
specifically to work around the `CUDA::cuda_driver` CMake link failure
documented in `README.md` §4 — it is a **build-compatibility fix, not a
performance tuning choice**. Re-enabling VMM to chase a possible
allocator-efficiency gain is out of scope unless the underlying CMake
link issue is independently resolved first; touching it risks
reintroducing the original build failure.

## Candidate Optimizations

### Candidate 1 — Confirm DFlash block_size and variant — RESOLVED (block_size), PARTIALLY OPEN (variant)

Source evidence: `common/speculative.cpp:963-995`, `src/models/dflash.cpp:156-158`
(exact `LOG_INF` lines that print `block_size`, `is_dflash2`-relevant
tensor info).

**Update: `dflash.block_size = 16` is now EXTERNALLY VERIFIED** (see
DFlash Analysis section above and `optimize.md`) — confirmed outside this
repo's own artifacts, not recovered from any log/doc that existed here.
This resolves the block_size half of this candidate: `--spec-draft-n-max
15` is confirmed to be the maximum valid value, and the prepared sweep
(`8,10,12,14,15`) is confirmed to cover the entire valid range.

**Still open:** whether Muse Glimmer's drafter is DFlash1, DFlash2
(selector variant, adds CPU-side lattice walk), or DSpark is **not**
resolved by the `block_size` fact alone — that would need the separate
`is_dflash2`/selector-tensor log line from an actual run. Lower priority
now that the range question is settled, but still worth checking
opportunistically if/when a real run's console output is captured.

Potential bottleneck: N/A — this is a diagnostic step, not a performance
change.

Why it may matter: The block_size half is now resolved (see above); the
remaining DFlash-variant question affects whether extra CPU-side selector
overhead (DFlash2 only) is a factor in our results.

Risk: None.

Required code changes: None.

Required benchmark: None — check saved console logs from the original
17.9 tok/s run first; if unavailable, the variant question can be
answered opportunistically from the stdout/stderr of the draft-length
sweep once it's run (no dedicated run needed just for this).

Success criterion: N/A (informational).

Status: **block_size — CONFIRMED (externally verified). DFlash
variant (1 vs. 2 vs. DSpark) — still NOT TESTED.**

### Candidate 2 — `GGML_CUDA_DISABLE_GRAPHS=1` diagnostic run

Source evidence: `ggml/src/ggml-cuda/common.cuh:1288-1291` (the actual
env var), `ggml-cuda.cu:4436-4493` (warmup/capture mechanism).

Potential bottleneck: If DFlash's dual-mode, variable-shape graphs
prevent CUDA graph capture from ever reaching steady "replay" state, the
capture/warmup overhead may be a net negative sunk cost relative to
direct kernel launches.

Why it may matter specifically on 2× T4: Not GPU-count-specific, but T4
IS eligible for CUDA graphs (`cc=750 >= Volta`), so this mechanism is
live on our hardware, unlike e.g. an older Pascal card where graphs would
already be disabled by the arch check.

Risk: None — reversible env var, no code/build change, doesn't touch
model/quantization/DFlash configuration.

Required code changes: None.

Required benchmark: Same exact DFlash command as the current best
(`--spec-draft-n-max 15`, everything else identical), once with
`GGML_CUDA_DISABLE_GRAPHS` unset (control) and once with it set to `1`.

Success criterion: Generation tok/s with graphs disabled vs. enabled —
if disabling graphs is *faster*, that's strong evidence the graph-shape
volatility hypothesis is real and worth pursuing further; if it's
*slower*, graphs are already helping and this line of inquiry ends.

Status: NOT TESTED.

### Candidate 3 — DFlash draft-length sweep below 15 (already prepared)

Source evidence: `common/speculative.cpp:997-1005` (clamping logic).

Potential bottleneck: A shorter draft block means less compute wasted on
draft positions that get rejected anyway, at the cost of less
amortization per accepted block.

Why it may matter specifically on 2× T4: Not hardware-specific — this is
a speculative-decoding acceptance-rate tradeoff, independent of GPU
architecture.

Risk: Low — already-supported CLI flag, no rebuild.

Required code changes: None — harness already exists
(`sweep_draft_length.py`).

Required benchmark: The prepared sweep itself.

Success criterion: Any value's generation tok/s exceeding 17.9,
confirmed by a second identical run (per the sweep's built-in
verification-before-declaring-a-new-best logic).

Status: NOT TESTED (harness ready).

### Candidate 4 — `--split-mode tensor` at the current best DFlash config

Source evidence: `docs/multi-gpu.md` (fetched 2026-09-28): *"pipeline-parallel
maximizes batch throughput; tensor-parallel minimizes latency."*

Potential bottleneck: Default `layer` (pipeline) split is
throughput-oriented; our workload is single-stream decode-latency-bound,
which is exactly what `tensor` split is documented to target instead.

Why it may matter specifically on 2× T4: Tensor split requires
significant cross-GPU communication per layer (via NCCL, per
`docs/multi-gpu.md`); T4 pairs on Kaggle commonly lack NVLink and share
only PCIe, which could make the added communication cost outweigh the
latency benefit — genuinely unknown without testing.

Risk: Medium — documented as "experimental"; could be unstable or simply
slower. Should be tested as an isolated A/B, reverted immediately if
worse or unstable.

Required code changes: None — CLI flag only (`--split-mode tensor` /
`-sm tensor`).

Required benchmark: Identical DFlash command, only `--split-mode`
changed (default vs `tensor`).

Success criterion: Generation tok/s improvement over 17.9, verified with
a second run; also check for correctness (garbled output would indicate
the experimental mode misbehaving with this model/DFlash combination).

Status: NOT TESTED.

### Candidate 5 — Manual `--tensor-split` rebalancing

Source evidence: `docs/multi-gpu.md`.

Potential bottleneck: Automatic layer split is memory-proportional, not
compute-proportional; with non-uniform per-layer compute this could
strand one GPU as a straggler.

Why it may matter specifically on 2× T4: With two identical-VRAM T4s, the
automatic split is likely already even — this candidate is only
worth pursuing if GPU utilization monitoring (`nvidia-smi`) during a run
shows a persistent imbalance.

Risk: None — CLI flag only.

Required code changes: None.

Required benchmark: Only worth running after observing an actual
imbalance; not a standalone recommended first step.

Success criterion: Generation tok/s improvement after rebalancing toward
the observed straggler GPU.

Status: NOT TESTED. Gated on an observation (GPU utilization check) we
don't have yet.

### Candidate 6 — `GGML_CUDA_GRAPH_OPT=1` (extra stream-parallel graph optimization)

Source evidence: `ggml-cuda.cu:4621-4628`.

Potential bottleneck: N/A directly — this is an opt-in additional
optimization pass over an already-captured graph, not a fix for a known
problem.

Why it may matter specifically on 2× T4: Not hardware-specific.

Risk: Low — env var, reversible, but off-by-default and undocumented
elsewhere, suggesting it may be less mature/tested than the base graph
path.

Required code changes: None.

Required benchmark: Only meaningful *after* Candidate 2 confirms CUDA
graphs are actually reaching steady replay state for DFlash's workload —
if graphs aren't stabilizing at all, this optimization pass has nothing
to act on.

Success criterion: Generation tok/s improvement over the Candidate-2
graphs-enabled baseline.

Status: NOT TESTED. Sequenced after Candidate 2.

### Candidate 7 — T4/Turing-specific MMQ kernel tuning table

Source evidence: `mmq.cuh:256-257` (no Turing-specific config; T4 shares
Ampere's tuning table).

Potential bottleneck: Ampere-tuned tile sizes/occupancy/thread counts may
not be optimal for Turing's different shared-memory-per-SM and lack of
`cp.async`.

Why it may matter specifically on 2× T4: Directly — this is the one
candidate that is T4-architecture-specific by construction.

Risk: **High** — requires writing and empirically tuning new CUDA kernel
launch-configuration code, then validating numerical correctness (a
kernel tuning bug can silently corrupt output, not just run slower).
This is upstream-contribution-grade work.

Required code changes: **Yes — requires modifying/adding to
`ggml-cuda`'s MMQ config headers and rebuilding llama.cpp from source.**

Required benchmark: Full before/after decode tok/s comparison plus
correctness validation (logit/output comparison against the existing
Ampere-table baseline) — well beyond a simple flag A/B.

Success criterion: Measurable tok/s improvement with no correctness
regression, validated across multiple runs.

Status: NOT TESTED. **Not recommended for near-term Kaggle
experimentation** — see "Things We Should NOT Change."

### Candidate 8 — Reduce DFlash's CPU-mediated target→draft feature marshaling

Source evidence: `common/speculative.cpp:1088-1178` (`process()`'s
per-token, per-layer `memcpy` loop into `batch_inject`, followed by a
dedicated `llama_decode(ctx_dft, ...)` call).

Potential bottleneck: CPU-side data marshaling and an extra decode call
on every target step, not just at DFlash block boundaries.

Why it may matter specifically on 2× T4: Not GPU-architecture-specific,
but relatively more significant on GPUs (like T4) with less raw compute
throughput, where a fixed CPU-overhead cost per step is a larger
fraction of total step time.

Risk: **High** — requires modifying `common/speculative.cpp`'s core
DFlash implementation and rebuilding; unlike the EAGLE3 implementation in
the same file, there is no existing upstream TODO/comment suggesting this
specific path for DFlash, so this would be original engineering, not
applying a documented fix.

Required code changes: **Yes — C++ changes to `common/speculative.cpp`
plus a llama.cpp rebuild.**

Required benchmark: Requires profiling (not just a tok/s before/after)
to first confirm this is actually a measurable fraction of decode time
before investing in a fix.

Success criterion: Profiled CPU-marshaling time reduced, with a
corresponding measured tok/s improvement and no correctness regression.

Status: NOT TESTED. **Not recommended for near-term Kaggle
experimentation** — requires profiling infrastructure we don't have set
up, and is high-risk source modification.

## Recommended Experiment Order

Based on technical evidence above, ordered by (cost, risk, information
value) — cheapest and safest first:

1. **Candidate 1 — block_size half is DONE.** `dflash.block_size = 16` is
   externally verified; the sweep is confirmed to cover the full valid
   range. The DFlash-variant half (DFlash1 vs. DFlash2 vs. DSpark) is
   still open but low-priority — check opportunistically from the sweep's
   own console output rather than as a separate step.
2. **Candidate 3 — run the already-prepared draft-length sweep**
   (`8,10,12,14,15`). This is now the immediate next experiment: zero new
   engineering, matches the existing controlled-optimization methodology
   in `optimize.md`, and covers the drafter's entire valid range.
3. **Candidate 2** — `GGML_CUDA_DISABLE_GRAPHS=1` diagnostic at the
   current best config. Trivial, zero rebuild, directly tests the
   CUDA-graph-shape-volatility hypothesis from source.
4. **Candidate 6** — `GGML_CUDA_GRAPH_OPT=1`, only if Candidate 2 shows
   graphs are net-positive and reaching steady state.
5. **Candidate 4** — `--split-mode tensor` A/B at the current best
   config. Trivial flag, but "experimental" — treat as a genuine
   coin-flip experiment, not an expected win, and check output
   correctness, not just speed.
6. **Candidate 5** — manual `--tensor-split`, only if GPU utilization
   monitoring during any of the above reveals a persistent imbalance.
7. **Candidates 7 and 8** — deprioritized. Both require modifying and
   rebuilding llama.cpp's actual source, carry real correctness risk,
   and (for Candidate 8 especially) need profiling infrastructure we
   don't have. Revisit only after 1–6 are exhausted and if there's
   appetite for a much larger engineering investment.

## Things We Should NOT Change

Already tested, do not re-litigate without new evidence:

| Configuration | Generation tok/s | Verdict |
|---|---:|---|
| Baseline (no DFlash) | 13.7 | Reference point only |
| `-b 512` | 13.6 | No decode improvement — tested, not adopted |
| Flash Attention (`-fa on`) | 13.7 | No decode improvement — tested, not adopted |
| DFlash (`--spec-draft-n-max 15`) | **17.9** | **Current best — do not remove** |

Additional guidance from this research pass:

- **Do not change the model or quantization** (explicit constraint for
  this phase; also, Q4_K_M's kernel path is shared with Ampere's tuning
  table regardless of quantization choice — switching quantization
  wouldn't touch the T4/Turing tuning gap identified above anyway).
- **Do not remove DFlash** (explicit constraint; it's the only lever that
  has actually moved decode tok/s so far).
- **Do not re-enable CUDA VMM** (`GGML_CUDA_NO_VMM`) to chase a
  hypothetical allocator speedup — it was adopted specifically to work
  around the documented `CUDA::cuda_driver` CMake link failure; touching
  it risks reintroducing that build failure for no confirmed benefit.
- **Do not test `--spec-draft-n-max` values above 15** — `dflash.block_size
  = 16` is externally verified, so the value is clamped to `block_size - 1
  = 15` at runtime (with a warning) for any higher request. 15 is
  confirmed to be the ceiling, not a guess; a ">15" experiment would
  silently just re-run n_max=15 under a different label. Do not pass
  `--allow-above-15` to the sweep script.
- **Do not attempt the T4-specific MMQ kernel rewrite (Candidate 7) or
  the DFlash CPU-marshaling reduction (Candidate 8) without first
  exhausting the zero-rebuild candidates (1–6)** — both require modifying
  and rebuilding llama.cpp's actual source with real correctness risk,
  for an unverified payoff.
- **Do not compare any future optimized-but-DFlash-disabled configuration
  against 17.9 tok/s** and call it a regression — that comparison is
  invalid by construction (see `optimize.md`).
