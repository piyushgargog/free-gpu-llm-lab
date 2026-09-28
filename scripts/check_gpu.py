#!/usr/bin/env python
"""Print visible GPU(s), or confirm none are visible.

Safe to run anywhere — only queries hardware, never loads model weights.

Usage:
    python scripts/check_gpu.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from free_gpu_llm.hardware import describe_hardware, get_gpu_info, is_cuda_available  # noqa: E402


def main() -> None:
    print(f"CUDA available: {is_cuda_available()}")
    print(f"Summary: {describe_hardware()}")
    gpus = get_gpu_info()
    if not gpus:
        print("No CUDA GPUs detected.")
        return
    for gpu in gpus:
        mem = f"{gpu.total_memory_gb} GB" if gpu.total_memory_gb is not None else "unknown"
        print(f"  [{gpu.index}] {gpu.name} — {mem}")


if __name__ == "__main__":
    main()
