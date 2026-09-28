#!/usr/bin/env python
"""Gemma 3 4B conversational Gradio UI — no web retrieval.

*** REMOTE GPU ONLY (Colab/Kaggle). Never run on a personal PC. ***

The model is loaded ONCE at module import time, not per message. For the
web-augmented version, see app/gradio_app.py.

Usage (in Colab):
    python experiments/gemma3-4b/gradio_app.py
"""

from __future__ import annotations

from free_gpu_llm.config import GenerationConfig, ModelConfig, get_hf_token
from free_gpu_llm.generation import build_messages, generate, load_model_and_tokenizer

REPO_ID = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are Gemma 3 4B, an AI assistant running locally on a Google Colab "
    "NVIDIA T4 GPU. You are not Gemini. Answer accurately and concisely. "
    "If you don't know something, say so rather than inventing information."
)

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        model_config = ModelConfig(name="gemma-3-4b-it-bf16", repo_id=REPO_ID, dtype="bfloat16", attn_implementation="sdpa")
        _model, _tokenizer = load_model_and_tokenizer(model_config, hf_token=get_hf_token())
    return _model, _tokenizer


def respond(message: str, history: list) -> str:
    model, tokenizer = _get_model()
    messages = build_messages(SYSTEM_PROMPT, history, message)
    return generate(model, tokenizer, messages, GenerationConfig(max_new_tokens=256))


def main() -> None:
    import gradio as gr

    _get_model()  # load once, before the server starts accepting traffic

    demo = gr.ChatInterface(
        fn=respond,
        title="Gemma 3 4B (Colab T4)",
        description="Running locally on a Colab T4 GPU. No web access.",
    )
    demo.launch()


if __name__ == "__main__":
    main()
