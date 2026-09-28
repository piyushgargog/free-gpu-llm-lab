## What does this PR change?

<!-- Short description. Link any related issue. -->

## Checklist

- [ ] I ran `python -m compileall src experiments scripts app tests` and `pytest tests/` locally, and both pass.
- [ ] Documentation was updated if this changes behavior, results, or setup steps.
- [ ] No model weights (`.gguf`, `.safetensors`, `.bin`, `.pt`, `.ckpt`, etc.) are included in this PR.
- [ ] No API keys, tokens, or `.env` files are included in this PR.
- [ ] Any benchmark/performance claim in this PR is backed by an actual measurement I ran myself — not estimated or assumed.
- [ ] Results are clearly labeled as **measured**, **externally verified**, **inferred/hypothesized**, or **not yet run** (see `CONTRIBUTING.md`).

## If this PR adds or changes an experiment

- [ ] Reproducibility info is included: hardware, exact command/config, software versions.
- [ ] New scripts that load/download a model or run inference are clearly marked `*** REMOTE GPU ONLY ***` and are not runnable by accident locally.
- [ ] `results.jsonl` entries were appended, not edited/deleted.

## Anything else reviewers should know?
