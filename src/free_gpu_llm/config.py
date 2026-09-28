"""Configuration dataclasses shared by experiment scripts.

Pure data structures only — no GPU/model imports here, so this module is
always safe to import on a local machine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def load_dotenv_if_present() -> None:
    """Load a local .env file if python-dotenv is available and a .env exists.

    Never raises — missing dependency or missing file are both fine.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)


def get_hf_token() -> Optional[str]:
    """Read HF_TOKEN from the environment. Never hardcode tokens in source."""
    load_dotenv_if_present()
    return os.environ.get("HF_TOKEN") or None


@dataclass
class GenerationConfig:
    """Decoding parameters for a single generation run."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True


@dataclass
class ModelConfig:
    """Identifies a model + how it should be loaded.

    `dtype` and `quantization` are informational/config fields; the actual
    torch dtype object is resolved lazily at load time in generation.py so
    that importing this module never requires torch to be installed.
    """

    name: str
    repo_id: str
    dtype: str = "bfloat16"  # "float16" | "bfloat16" | "nf4" | "gguf"
    quantization: Optional[str] = None  # e.g. "NF4", "Q4_0", "Q4_K_M", "GPTQ-Int4"
    context_length: int = 2048
    attn_implementation: str = "sdpa"
    local_files_only: bool = False
    extra: dict = field(default_factory=dict)
