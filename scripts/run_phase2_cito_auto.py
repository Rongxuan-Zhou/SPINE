#!/usr/bin/env python
"""Run Phase 2 tuning-free CITO in sequential batches."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto-run Phase2 CITO in batches.")
    parser.add_argument("--input-hdf5", type=Path, required=True)
    parser.add_argument("--source-hdf5", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=str, default="refine")
    parser.add_argument("--suffix", type=str, default="tune_c")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=None)
    parser.add_argument("--total", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--demo-timeout", type=float, default=120)
    parser.add_argument("--scvx-iters", type=int, default=1)
    parser.add_argument("--penalty-loops", type=int, default=2)
    parser.add_argument("--trust-region", type=float, default=0.2)
    parser.add_argument("--solver", type=str, default=None)
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument("--merge-output", type=Path, default=None)
    return parser.parse_args()


def _build_output_path(
    output_dir: Path, prefix: str, suffix: str, batch_end: int
) -> Path:
    name = f"{prefix}_{batch_end}_{suffix}.hdf5"
    return output_dir / name


def _run_batch(args: argparse.Namespace, output_path: Path, start: int, limit: int) -> None:
    cmd: List[str] = [
        "python",
        "scripts/run_phase2_cito_batch.py",
        "--input-hdf5",
        str(args.input_hdf5),
        "--source-hdf5",
        str(args.source_hdf5),
        "--output-hdf5",
        str(output_path),
        "--limit",
        str(limit),
        "--start",
        str(start),
        "--demo-timeout",
        str(args.demo_timeout),
        "--scvx-iters",
        str(args.scvx_iters),
        "--penalty-loops",
        str(args.penalty_loops),
        "--trust-region",
        str(args.trust_region),
    ]
    if args.allow_fail:
        cmd.append("--allow-fail")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.solver:
        cmd.extend(["--solver", args.solver])
    if args.solver_verbose:
        cmd.append("--solver-verbose")

    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def _merge_outputs(output: Path, inputs: List[Path]) -> None:
    cmd = [
        "python",
        "scripts/merge_hdf5_demos.py",
        "--output-hdf5",
        str(output),
        "--inputs",
        *[str(p) for p in inputs],
    ]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=False)


def main() -> None:
    args = _parse_args()
    if args.num_batches is None and args.total is None:
        raise ValueError("Either --num-batches or --total must be set.")

    if args.total is not None:
        num_batches = int(math.ceil(args.total / args.batch_size))
    else:
        num_batches = int(args.num_batches)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []

    for i in range(num_batches):
        batch_start = args.start + i * args.batch_size
        batch_end = batch_start + args.batch_size
        output_path = _build_output_path(
            args.output_dir, args.output_prefix, args.suffix, batch_end
        )
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists", flush=True)
            outputs.append(output_path)
            continue
        _run_batch(args, output_path, batch_start, args.batch_size)
        outputs.append(output_path)

    if args.merge_output:
        _merge_outputs(args.merge_output, outputs)


if __name__ == "__main__":
    main()
