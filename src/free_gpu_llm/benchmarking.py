"""Benchmark result recording + llama.cpp output parsing.

Pure data/text processing — no GPU or model imports, so this module (and its
tests) run fine on a local machine with no CUDA.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class BenchmarkResult:
    """A single benchmark measurement.

    `valid` must be False for any run with known numerical issues (e.g. NaN
    logits) or that wasn't actually completed. Invalid runs are still
    recorded (for the historical record) but must never be reported as a
    usable performance number.
    """

    model: str
    hardware: str
    tokens_per_second: float
    model_path: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_name: Optional[str] = None
    quantization: Optional[str] = None
    dtype: Optional[str] = None
    context_length: Optional[int] = None
    batch_size: Optional[int] = None
    speculative_decoding: Optional[str] = None
    prompt: Optional[str] = None
    prompt_tokens: Optional[int] = None
    generated_tokens: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    prompt_tokens_per_second: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    software_version: Optional[str] = None
    git_commit: Optional[str] = None
    valid: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def tokens_per_second(token_count: int, elapsed_seconds: float) -> float:
    """Compute a tok/s rate. Raises on non-positive elapsed time rather than
    silently returning inf/garbage."""
    if elapsed_seconds <= 0:
        raise ValueError(f"elapsed_seconds must be > 0, got {elapsed_seconds}")
    return token_count / elapsed_seconds


def get_git_commit(repo_path: str | Path = ".") -> Optional[str]:
    """Best-effort short git commit hash for provenance. Returns None if
    unavailable (e.g. not a git repo) rather than raising."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def append_result(result: BenchmarkResult, path: str | Path) -> None:
    """Append one result as a line of JSONL. Never overwrites prior results —
    each call adds a new line to the historical record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")


def load_results(path: str | Path) -> list[dict]:
    """Load all recorded results from a JSONL file. Returns [] if the file
    doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    results = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


_TOKENS_PER_SECOND_RE = re.compile(r"([\d.]+)\s*tokens per second")


def parse_llama_cpp_timings(output: str) -> dict[str, Optional[float]]:
    """Parse tok/s figures out of llama.cpp (llama-cli) stderr/stdout timing
    output, e.g. lines like:

        llama_print_timings: prompt eval time =  xxx ms /  N tokens ( x.xx ms per token, 104.7 tokens per second)
        llama_print_timings:        eval time =  xxx ms /  N runs   ( x.xx ms per token,  13.7 tokens per second)

    Returns {"prompt_tokens_per_second": float|None, "eval_tokens_per_second": float|None}.
    Missing lines simply leave the corresponding value as None — this parser
    never fabricates a number that wasn't in the text.
    """
    result: dict[str, Optional[float]] = {
        "prompt_tokens_per_second": None,
        "eval_tokens_per_second": None,
    }
    for line in output.splitlines():
        stripped = line.strip()
        if "prompt eval time" in stripped:
            m = _TOKENS_PER_SECOND_RE.search(stripped)
            if m:
                result["prompt_tokens_per_second"] = float(m.group(1))
        elif "eval time" in stripped and "prompt" not in stripped and "sample" not in stripped:
            m = _TOKENS_PER_SECOND_RE.search(stripped)
            if m:
                result["eval_tokens_per_second"] = float(m.group(1))
    return result
