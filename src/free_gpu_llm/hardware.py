"""GPU/hardware detection utilities.

These functions only *query* hardware state (device names, memory totals).
They never download or load model weights, so they are safe to call on a
local machine — they'll simply report "no GPU" there.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class GPUInfo:
    index: int
    name: str
    total_memory_gb: Optional[float] = None


def get_gpu_info() -> "List[GPUInfo]":
    """Return info for visible CUDA GPUs, or an empty list if none / no torch."""
    try:
        import torch
    except ImportError:
        return _get_gpu_info_via_nvidia_smi()

    if not torch.cuda.is_available():
        return []

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            GPUInfo(
                index=i,
                name=props.name,
                total_memory_gb=round(props.total_memory / (1024**3), 2),
            )
        )
    return gpus


def _get_gpu_info_via_nvidia_smi() -> "List[GPUInfo]":
    """Fallback when torch isn't installed: shell out to nvidia-smi if present."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    if out.returncode != 0:
        return []

    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        idx, name, mem_mib = parts
        try:
            gpus.append(GPUInfo(index=int(idx), name=name, total_memory_gb=round(float(mem_mib) / 1024, 2)))
        except ValueError:
            continue
    return gpus


def is_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return bool(_get_gpu_info_via_nvidia_smi())
    return torch.cuda.is_available()


def describe_hardware() -> str:
    """Human-readable hardware summary, e.g. '2x Tesla T4' or 'CPU only (no GPU detected)'."""
    gpus = get_gpu_info()
    if not gpus:
        return "CPU only (no GPU detected)"

    names = {}
    for gpu in gpus:
        names[gpu.name] = names.get(gpu.name, 0) + 1

    parts = [f"{count}x {name}" for name, count in names.items()]
    return ", ".join(parts)
