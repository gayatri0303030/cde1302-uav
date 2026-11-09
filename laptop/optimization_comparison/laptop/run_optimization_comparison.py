#!/usr/bin/env python3
"""Run the bundled offline before/after calculation and print its results."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=root / "data" / "observations.zip")
    parser.add_argument("--output", type=Path,
                        default=root / "results" / f"run_{time.time_ns()}",
                        help="New output directory; existing results are never overwritten")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Recording not found: {args.archive}; supply --archive PATH")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new --output directory")
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    with tempfile.TemporaryDirectory(prefix="tag-comparison-mpl-") as mpl_cache:
        env["MPLCONFIGDIR"] = mpl_cache
        return subprocess.run([
            sys.executable, str(root / "laptop" / "analyze_observations.py"),
            str(args.archive.resolve()), "--output", str(args.output.resolve()),
        ], env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
