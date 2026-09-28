# Architecture

## Two environments — never mix them

```mermaid
flowchart TB
    subgraph LOCAL["PERSONAL PC (this repo's dev environment)"]
        L1[Write/edit code]
        L2[Write docs]
        L3[Lightweight tests\n- no GPU\n- no model weights\n- mocks/synthetic data]
        L4[Static checks / compileall]
        L5[Git operations]
    end

    subgraph REMOTE["REMOTE GPU (Colab / Kaggle)"]
        R1[Download model weights]
        R2[Load models]
        R3[Run inference]
        R4[Benchmark]
        R5[Build llama.cpp with CUDA]
    end

    LOCAL -- "code + scripts (not executed here)" --> REMOTE
    REMOTE -- "captured logs / results.jsonl" --> LOCAL
```

The personal PC **never** downloads, caches, loads, or runs LLM weights.
Every script under `experiments/*/run.py`, `experiments/*/benchmark.py`,
`app/gradio_app.py`, and `scripts/download_model.py` is written to be
executed remotely — see the `*** REMOTE GPU ONLY ***` banner at the top of
each such file.

## Repository layers

```mermaid
flowchart TB
    subgraph experiments["experiments/<name>/"]
        E1[README.md - what happened, real numbers]
        E2[run.py / benchmark.py - remote-only scripts]
        E3[results.jsonl - append-only benchmark records]
    end

    subgraph src["src/free_gpu_llm/"]
        S1[config.py - dataclasses, env loading]
        S2[hardware.py - GPU detection]
        S3[generation.py - chat history, logit validation,\nmodel load/generate (lazy torch import)]
        S4[benchmarking.py - BenchmarkResult, JSONL I/O,\nllama.cpp output parsing]
        S5[web/ - search, extract, retrieval, pipeline]
    end

    subgraph apps["app/, scripts/"]
        A1[app/gradio_app.py - web-augmented chatbot]
        A2[scripts/check_environment.py, check_gpu.py]
        A3[scripts/benchmark.py, download_model.py]
    end

    experiments --> src
    apps --> src
```

`src/free_gpu_llm` is the shared library. Every module that could require a
GPU or model weights (`generation.load_model_and_tokenizer`, `generate`,
`web.search.search`, `web.extract.extract_text`) imports its heavy
dependency **lazily, inside the function**, specifically so that:

1. The package imports cleanly on a machine with no `torch`/`transformers`/
   `llama_cpp`/`gradio`/`ddgs`/`trafilatura` installed.
2. `tests/` can exercise the pure-logic parts (history normalization, NaN/Inf
   checks, URL dedup, tok/s math, llama.cpp log parsing) without a GPU.

## Benchmark data flow

```mermaid
sequenceDiagram
    participant Remote as Remote GPU (Colab/Kaggle)
    participant Script as experiment benchmark.py
    participant Lib as free_gpu_llm.benchmarking
    participant File as results.jsonl

    Remote->>Script: run llama-cli / model.generate()
    Script->>Script: time it, capture output
    Script->>Lib: parse_llama_cpp_timings() or validate_logits()
    Lib-->>Script: tok/s, valid: bool
    Script->>Lib: BenchmarkResult(...)
    Lib->>File: append_result() — appends, never overwrites
```

Each `results.jsonl` is append-only (`free_gpu_llm.benchmarking.append_result`)
specifically so the historical record — including invalid/failed runs — is
preserved, not silently replaced by the latest run.

## Experiment timeline

```mermaid
flowchart LR
    Q[Qwen3-30B-A3B\nGPTQ, ~1.8 tok/s] --> G1[Gemma 3 4B\nFP16 -> NaN]
    G1 --> G2[Gemma 3 4B\nBF16 ~10.04 tok/s]
    G2 --> G3[Gemma 3 4B\nNF4 ~7.84 tok/s]
    G3 --> UI[Gradio chatbot]
    UI --> WEB[DDGS + trafilatura\nweb retrieval]
    WEB --> G4[Gemma 4 E4B\nQ4_0 GGUF, llama.cpp\nCUDA build - IN PROGRESS]
    G4 --> M1[Muse Glimmer 30B\nllama.cpp native CUDA\nbaseline ~13.7 tok/s]
    M1 --> M2[2x T4 optimization\nbatch/flash-attn: no gain]
    M2 --> M3[DFlash speculative decoding\n~17.9 tok/s - best observed]
```
