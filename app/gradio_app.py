#!/usr/bin/env python
"""Web-augmented Gemma 3 4B chatbot (DDGS + trafilatura retrieval).

*** REMOTE GPU ONLY (Colab/Kaggle). Never run on a personal PC. ***

Architecture:

    User -> Gradio -> should_search() -> DDGS search -> trafilatura extract
          -> retrieved context -> Gemma -> answer

This is external retrieval + prompt augmentation, NOT autonomous browsing
and NOT native model tool-calling. See docs/web-retrieval.md. The model is
loaded ONCE at startup, not per message.

Usage (in Colab):
    python app/gradio_app.py
"""

from __future__ import annotations

from free_gpu_llm.config import GenerationConfig, ModelConfig, get_hf_token
from free_gpu_llm.generation import build_messages, generate, load_model_and_tokenizer
from free_gpu_llm.web.pipeline import build_augmented_context

REPO_ID = "google/gemma-3-4b-it"

SYSTEM_PROMPT = (
    "You are Gemma 3 4B, an AI assistant running locally on a Google Colab "
    "NVIDIA T4 GPU. You are not Gemini. Answer accurately and concisely. "
    "If you don't know something, say so rather than inventing information. "
    "When web search context is provided below a user message, prefer it "
    "over your own prior knowledge for facts that may have changed, and "
    "cite it informally (e.g. 'according to the search results...')."
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

    context = build_augmented_context(message)
    if context:
        augmented_message = f"Web search context:\n\n{context}\n\nUser question: {message}"
    else:
        augmented_message = message

    messages = build_messages(SYSTEM_PROMPT, history, augmented_message)
    return generate(model, tokenizer, messages, GenerationConfig(max_new_tokens=256))


def main() -> None:
    import gradio as gr

    _get_model()  # load once, before the server starts accepting traffic

    demo = gr.ChatInterface(
        fn=respond,
        title="Gemma 3 4B + Web Retrieval (Colab T4)",
        description=(
            "Running locally on a Colab T4 GPU. Questions containing words like "
            "'latest', 'news', 'today', 'price', 'weather' trigger a DuckDuckGo "
            "search + page extraction before answering (see free_gpu_llm.web.pipeline)."
        ),
    )
    demo.launch()


if __name__ == "__main__":
    main()
