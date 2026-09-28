"""Chat-history handling and numerical validation helpers.

The pure-python pieces here (history normalization, logit validation on
plain float iterables) have no GPU/model dependency and are unit-tested
locally. `load_model_and_tokenizer` / `generate` DO import torch/transformers
(lazily) and are only ever meant to be called in a remote GPU environment —
they are not exercised by the local test suite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional

from free_gpu_llm.config import GenerationConfig, ModelConfig


@dataclass
class ChatTurn:
    role: str  # "user" | "assistant" | "system"
    content: str


def normalize_history(history: Any) -> list[ChatTurn]:
    """Normalize a chat history into a clean list of ChatTurn.

    Accepts:
      - a list of (user, assistant) tuples/lists (Gradio's legacy format),
        where `assistant` may be None for an in-flight turn
      - a list of {"role": ..., "content": ...} dicts (OpenAI-style)
      - None / [] -> []

    Drops entries with empty/whitespace-only or non-string content instead
    of raising, since chat UIs routinely produce partial/malformed turns.
    """
    if not history:
        return []

    turns: list[ChatTurn] = []
    for entry in history:
        if isinstance(entry, dict):
            role = entry.get("role")
            content = entry.get("content")
            if _is_usable_text(content) and role in ("user", "assistant", "system"):
                turns.append(ChatTurn(role=role, content=content.strip()))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            user_msg, assistant_msg = entry
            if _is_usable_text(user_msg):
                turns.append(ChatTurn(role="user", content=user_msg.strip()))
            if _is_usable_text(assistant_msg):
                turns.append(ChatTurn(role="assistant", content=assistant_msg.strip()))
        # Anything else (wrong shape) is silently dropped rather than raising,
        # since a single malformed turn shouldn't crash the whole chatbot.
    return turns


def _is_usable_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def build_messages(system_prompt: Optional[str], history: Any, user_message: str) -> list[dict]:
    """Build an OpenAI/HF-chat-template-style message list from a system
    prompt, prior history, and the new user message."""
    messages: list[dict] = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    for turn in normalize_history(history):
        messages.append({"role": turn.role, "content": turn.content})
    if _is_usable_text(user_message):
        messages.append({"role": "user", "content": user_message.strip()})
    return messages


@dataclass
class LogitValidation:
    total: int
    nan_count: int
    inf_count: int

    @property
    def valid(self) -> bool:
        return self.nan_count == 0 and self.inf_count == 0


def validate_logits(values: Iterable[float]) -> LogitValidation:
    """Check a flat iterable of logit values for NaN/Inf.

    Accepts plain floats, numpy arrays, or torch tensors (anything iterable
    of numbers, or anything with `.flatten().tolist()`). Returns a
    LogitValidation whose `.valid` is False if ANY NaN or Inf was found —
    such a result must never be reported as a valid benchmark
    (see the Gemma 3 FP16 case in docs/numerical-stability.md).
    """
    if hasattr(values, "flatten") and hasattr(values, "tolist"):
        values = values.flatten().tolist()

    total = 0
    nan_count = 0
    inf_count = 0
    for v in values:
        total += 1
        fv = float(v)
        if fv != fv:  # NaN check without requiring math/numpy import
            nan_count += 1
        elif fv in (float("inf"), float("-inf")):
            inf_count += 1
    return LogitValidation(total=total, nan_count=nan_count, inf_count=inf_count)


def resolve_torch_dtype(dtype_name: str):
    """Map a config dtype string to a torch dtype. Imports torch lazily —
    only called from remote-GPU code paths."""
    import torch

    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(
            f"Unsupported dtype '{dtype_name}'. Use one of: {sorted(mapping)}. "
            "For 4-bit, use ModelConfig(dtype='nf4') and load_model_and_tokenizer's "
            "bitsandbytes path instead of resolve_torch_dtype."
        )
    return mapping[dtype_name]


def load_model_and_tokenizer(config: ModelConfig, hf_token: Optional[str] = None):
    """Load a HF transformers model + tokenizer for the given config.

    REMOTE GPU ONLY. This function downloads/loads real model weights and
    must never be called on a machine that should not fetch LLM weights.
    Intentionally not covered by the local test suite.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config.repo_id, token=hf_token, local_files_only=config.local_files_only
    )

    load_kwargs: dict[str, Any] = {
        "token": hf_token,
        "local_files_only": config.local_files_only,
        "attn_implementation": config.attn_implementation,
        "device_map": "auto",
    }

    if config.dtype == "nf4":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        load_kwargs["dtype"] = resolve_torch_dtype(config.dtype)

    model = AutoModelForCausalLM.from_pretrained(config.repo_id, **load_kwargs)
    return model, tokenizer


def generate(model, tokenizer, messages: list[dict], gen_config: GenerationConfig) -> str:
    """Run one generation pass and return the decoded completion text.

    REMOTE GPU ONLY — operates on an already-loaded model/tokenizer.
    """
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=gen_config.max_new_tokens,
        do_sample=gen_config.do_sample,
        temperature=gen_config.temperature,
        top_p=gen_config.top_p,
        top_k=gen_config.top_k,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
