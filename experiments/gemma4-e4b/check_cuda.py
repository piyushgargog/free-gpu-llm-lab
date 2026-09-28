#!/usr/bin/env python
"""Check whether the installed llama-cpp-python build supports GPU offload.

Safe to run anywhere. On a personal PC (no CUDA build installed) this will
correctly report False — that is expected, not an error.

Usage:
    python experiments/gemma4-e4b/check_cuda.py
"""

from __future__ import annotations


def main() -> None:
    try:
        from llama_cpp import llama_cpp
    except ImportError:
        print("llama-cpp-python is not installed. Install it (with a CUDA build, remotely) to check GPU offload support.")
        return

    supported = llama_cpp.llama_supports_gpu_offload()
    print(f"llama_supports_gpu_offload(): {supported}")
    if not supported:
        print(
            "GPU offload NOT available with the current build. To fix (REMOTE ONLY):\n"
            "  pip uninstall -y llama-cpp-python\n"
            '  CMAKE_ARGS="-DGGML_CUDA=on" CMAKE_BUILD_PARALLEL_LEVEL=2 \\\n'
            "    pip install llama-cpp-python --no-cache-dir --force-reinstall"
        )


if __name__ == "__main__":
    main()
