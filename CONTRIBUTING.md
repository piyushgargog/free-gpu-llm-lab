# Contributing to Free GPU LLM Lab

Thanks for your interest. This is a research/experimentation lab
documenting real attempts to run and optimize open-weight LLMs on
free-tier GPU platforms (Google Colab, Kaggle). It is not a production
library — the priority is a **truthful, reproducible record**, not
feature velocity.

## What this repository is

- A collection of experiments (`experiments/<name>/`), each with its own
  README, scripts, and an append-only `results.jsonl` benchmark log.
- A shared library (`src/free_gpu_llm/`) used across experiments, with
  GPU/model dependencies imported lazily so it's testable without a GPU.
- Documentation (`docs/`) covering architecture, quantization,
  troubleshooting, and the full experiment history.

See `README.md` for the full picture and `docs/architecture.md` for how
the pieces fit together.

## Reporting bugs / issues

Open a GitHub issue using the **Bug report** template. Include what you
expected, what happened, and exact reproduction steps. If it's a local
(non-GPU) issue, a `python -m compileall .` / `pytest tests/` failure is
usually enough context; if it's a remote (Colab/Kaggle) issue, include
the environment (GPU, CUDA version, package versions) and, ideally, the
captured console output.

## Proposing a new LLM experiment

Open an issue using the **Experiment / benchmark report** template, or a
pull request that follows the existing experiment structure:

```
experiments/<your-experiment-name>/
├── README.md       # goal, config, exact commands, results
├── run.py / benchmark.py   # REMOTE GPU ONLY scripts
└── results.jsonl   # append-only benchmark records
```

Look at `experiments/muse-glimmer-30b/` for a fully worked example
(model, hardware, build config, baseline + optimization attempts, and a
controlled-optimization log in `optimize.md`).

## Reproducibility and benchmark honesty — the most important rule

This repository has one hard rule: **document reality, not aspiration.**

- **Never fabricate a benchmark number.** If you haven't run it, don't
  write a number for it — mark it `NOT RUN` / `NOT YET RUN` / `IN
  PROGRESS`.
- **Clearly distinguish four kinds of claims** in anything you write:
  - **Measured** — a number you personally benchmarked, with the exact
    command/config that produced it.
  - **Externally verified** — a fact confirmed from an authoritative
    external source (e.g. reading the actual llama.cpp source code for a
    flag's real behavior), not something you benchmarked yourself. Label
    it as such and cite the source.
  - **Inferred / hypothesized** — a plausible mechanism or expectation
    that has *not* been measured or externally confirmed. Say so
    explicitly (e.g. "hypothesis, not measured").
  - **Not yet run** — a prepared experiment/harness with no result yet.
- **A numerically invalid run (NaN/Inf logits, garbled output) is not a
  valid benchmark**, no matter how fast the wall-clock timer says it was.
  Validate before you report.
- **One variable at a time.** If you're proposing an optimization,
  change exactly one thing from the current best-known configuration and
  say what you compared against.
- Every `results.jsonl` is **append-only** — don't edit or delete
  historical entries; add new ones.

## Local-machine vs. remote-GPU safety

- **Never commit model weights** (`.gguf`, `.safetensors`, `.bin`, `.pt`,
  `.ckpt`, etc.) — `.gitignore` already excludes these; don't work around
  it.
- **Never commit API keys, tokens, or `.env` files** — only
  `.env.example` (with empty/placeholder values) belongs in the repo.
- Scripts that load/download models or run inference must be clearly
  marked `*** REMOTE GPU ONLY ***` in their docstring and must not be
  runnable by accident from a local machine's default code path.
- Free-GPU experiments should stay **reproducible where possible** — a
  precise command, exact software versions, and hardware description, so
  someone else with Colab/Kaggle access could re-run it.

## Before submitting a pull request

1. Run the local test suite (no GPU required):
   ```bash
   python -m compileall src experiments scripts app tests
   pytest tests/
   ```
2. If you touched documentation, check that any relative links you added
   actually resolve.
3. If you're adding a new experiment script that touches a model, make
   sure it's marked remote-only and does not execute anything on import.
4. Keep the PR focused — one experiment, one fix, or one doc change per
   PR is easier to review than a bundle of unrelated changes.

## Submitting the pull request

1. Fork the repo and create a branch from `master`.
2. Make your change, following the guidance above.
3. Fill out the PR template (tests run, no weights/secrets committed,
   benchmark claims backed by real measurements, reproducibility info for
   experiment changes).
4. Open the PR against `master`. CI (compile check + unit tests) runs
   automatically and must pass — it never requires a GPU or downloads
   anything.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
