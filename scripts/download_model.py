#!/usr/bin/env python
"""Template for downloading a model from Hugging Face Hub.

*** REMOTE GPU ENVIRONMENT ONLY (Colab / Kaggle). ***

This downloads multi-GB model weights. It must NEVER be run on a personal
PC that should not fetch/cache LLM weights. As a guard, it refuses to run
unless --confirm-remote-gpu is passed explicitly.

Usage (run this in a Colab/Kaggle notebook cell, not locally):

    python scripts/download_model.py \\
        --repo-id google/gemma-4-E4B-it-qat-q4_0-gguf \\
        --filename gemma-4-E4B_q4_0-it.gguf \\
        --local-dir /kaggle/working/models/gemma4-e4b \\
        --confirm-remote-gpu

    python scripts/download_model.py \\
        --repo-id google/gemma-3-4b-it \\
        --local-dir /content/models/gemma3-4b \\
        --confirm-remote-gpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. google/gemma-3-4b-it")
    parser.add_argument(
        "--filename",
        default=None,
        help="Single file to download (e.g. a .gguf file). Omit to download the full snapshot.",
    )
    parser.add_argument("--local-dir", required=True, help="Destination directory (on the remote filesystem)")
    parser.add_argument(
        "--confirm-remote-gpu",
        action="store_true",
        help="Required acknowledgement that this is running in Colab/Kaggle, not on a personal PC.",
    )
    args = parser.parse_args()

    if not args.confirm_remote_gpu:
        print(
            "Refusing to run.\n\n"
            "This script downloads multi-GB model weights and must ONLY be run in a\n"
            "remote GPU environment (Google Colab / Kaggle) — never on a personal PC.\n\n"
            "If you are certain you are in Colab/Kaggle right now, re-run with\n"
            "--confirm-remote-gpu.",
            file=sys.stderr,
        )
        sys.exit(1)

    from free_gpu_llm.config import get_hf_token

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("huggingface_hub is not installed. Install the 'gpu' extra in the remote environment.", file=sys.stderr)
        sys.exit(1)

    token = get_hf_token()
    Path(args.local_dir).mkdir(parents=True, exist_ok=True)

    if args.filename:
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=args.filename,
            local_dir=args.local_dir,
            token=token,
        )
    else:
        path = snapshot_download(
            repo_id=args.repo_id,
            local_dir=args.local_dir,
            token=token,
        )

    print(f"Downloaded to: {path}")


if __name__ == "__main__":
    main()
