"""free_gpu_llm: shared utilities for the free-gpu-llm-lab experiments.

Heavy/GPU dependencies (torch, transformers, bitsandbytes, llama-cpp-python)
are imported lazily inside individual functions so that this package can be
imported and unit-tested on a machine with no GPU and no model weights.
"""

__version__ = "0.1.0"
