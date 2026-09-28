#!/usr/bin/env python
"""Qwen3-30B-A3B (GPTQ Int4) — manual 2x T4 layer split.

*** REMOTE GPU ONLY (Kaggle, 2x Tesla T4). Never run on a personal PC. ***

STATUS: NOT VERIFIED. This is a best-effort reconstruction of the loading
approach used in the original (historical) experiment. The original model
weights were lost after a Kaggle session restart, so this script has not
been re-run end-to-end against the reconstructed code. Treat the exact
transformers/auto-gptq API calls here as a starting point to validate in a
fresh Kaggle session, not as a confirmed-working recipe.

Historical observed result for this approach: ~1.8 tokens/sec
(GPU0: ~28 layers, GPU1: ~20 layers). See README.md.
"""

from __future__ import annotations

import argparse

REPO_ID = "Qwen/Qwen3-30B-A3B-GPTQ-Int4"  # NOT VERIFIED exact repo id/tag used historically


def build_manual_device_map(n_layers_gpu0: int = 28, n_layers_gpu1: int = 20) -> dict:
    """Reconstruct the historical manual layer split: ~28 layers on GPU 0,
    ~20 layers on GPU 1. Non-layer modules (embeddings, norm, lm_head) are
    pinned to GPU 0 to avoid extra cross-device hops for the first/last op.

    NOT VERIFIED against the real Qwen3-30B-A3B module naming — the layer
    key prefix ("model.layers.{i}") matches standard HF Qwen3 architecture
    but should be confirmed against `model.config` / a printed module list
    in the remote environment before trusting this map.
    """
    device_map: dict = {
        "model.embed_tokens": 0,
        "model.norm": 0,
        "lm_head": 0,
    }
    total_layers = n_layers_gpu0 + n_layers_gpu1
    for i in range(total_layers):
        device_map[f"model.layers.{i}"] = 0 if i < n_layers_gpu0 else 1
    return device_map


def load_model_and_tokenizer(hf_token: str | None = None):
    """Load Qwen3-30B-A3B GPTQ with backend='torch' (Marlin kernel is
    incompatible with T4) and the manual per-GPU layer split above.

    REMOTE GPU ONLY. NOT VERIFIED end-to-end.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, token=hf_token)

    # backend="torch" avoids the Marlin kernel JIT/compat issue observed on T4.
    quant_config = GPTQConfig(bits=4, backend="torch")

    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        token=hf_token,
        quantization_config=quant_config,
        device_map=build_manual_device_map(),
        torch_dtype=torch.float16,
    )
    return model, tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompt", default="Explain what a transformer neural network is in simple technical terms.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    from free_gpu_llm.config import get_hf_token

    model, tokenizer = load_model_and_tokenizer(hf_token=get_hf_token())
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print(tokenizer.decode(output_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
