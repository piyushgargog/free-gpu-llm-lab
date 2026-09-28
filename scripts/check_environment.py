#!/usr/bin/env python
"""Report what's installed and what hardware is visible.

Safe to run ANYWHERE (local PC or remote GPU) — this only inspects the
environment. It never downloads or loads model weights.

Usage:
    python scripts/check_environment.py
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from free_gpu_llm.hardware import describe_hardware, get_gpu_info  # noqa: E402

OPTIONAL_PACKAGES = [
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "huggingface_hub",
    "llama_cpp",
    "gradio",
    "ddgs",
    "trafilatura",
]


def is_installed(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def main() -> None:
    print(f"Python: {platform.python_version()} ({platform.platform()})")
    print()

    print("Optional packages:")
    for pkg in OPTIONAL_PACKAGES:
        status = "installed" if is_installed(pkg) else "not installed"
        print(f"  {pkg:<16} {status}")
    print()

    print(f"Hardware: {describe_hardware()}")
    for gpu in get_gpu_info():
        mem = f"{gpu.total_memory_gb} GB" if gpu.total_memory_gb is not None else "unknown"
        print(f"  GPU {gpu.index}: {gpu.name} ({mem})")
    print()

    if not get_gpu_info():
        print(
            "No GPU detected. This is expected/fine on a personal PC - actual model\n"
            "experiments only happen in Colab/Kaggle. Do not attempt to download or\n"
            "load model weights in this environment."
        )


if __name__ == "__main__":
    main()
