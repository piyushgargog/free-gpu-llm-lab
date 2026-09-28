#!/usr/bin/env python
"""Download the Gemma 4 E4B Q4_0 GGUF file.

*** REMOTE GPU ONLY (Colab/Kaggle). NEVER run on a personal PC. ***
Downloads ~5.15 GB. Refuses to run without --confirm-remote-gpu.

Usage (in Colab/Kaggle):
    python experiments/gemma4-e4b/download.py \\
        --local-dir /kaggle/working/models/gemma4-e4b \\
        --confirm-remote-gpu
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ID = "google/gemma-4-E4B-it-qat-q4_0-gguf"
FILENAME = "gemma-4-E4B_q4_0-it.gguf"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--local-dir", required=True)
    parser.add_argument("--confirm-remote-gpu", action="store_true")
    args = parser.parse_args()

    if not args.confirm_remote_gpu:
        print(
            f"Refusing to run: this downloads ~5.15 GB ({FILENAME}) and must only\n"
            "run in Colab/Kaggle. Re-run with --confirm-remote-gpu if that's where you are.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Delegate to the shared downloader for the actual fetch.
    downloader = Path(__file__).resolve().parent.parent.parent / "scripts" / "download_model.py"
    subprocess.run(
        [
            sys.executable,
            str(downloader),
            "--repo-id", REPO_ID,
            "--filename", FILENAME,
            "--local-dir", args.local_dir,
            "--confirm-remote-gpu",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
