#!/usr/bin/env python
"""Auto-run single-demo CITO refinement for Threading and merge into a master HDF5."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import h5py


def _count_demos(path: Path) -> int:
    if not path.exists():
        return 0
    with h5py.File(path, "r") as f:
        return len(f["data"].keys())


def _load_demo_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with h5py.File(path, "r") as f:
        return set(f["data"].keys())


def _part_has_demo(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as f:
            return "data" in f and len(f["data"].keys()) > 0
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-run Threading refine batches.")
    parser.add_argument("--start", type=int, default=80)
    parser.add_argument("--target", type=int, default=200)
    parser.add_argument("--max-start", type=int, default=999)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--demo-timeout", type=int, default=60)
    parser.add_argument("--scvx-iters", type=int, default=1)
    parser.add_argument("--penalty-loops", type=int, default=2)
    parser.add_argument("--trust-region", type=float, default=0.2)
    parser.add_argument(
        "--input-hdf5",
        type=Path,
        default=Path("data/mimicgen_generated/threading/spine_threading/demo.hdf5"),
    )
    parser.add_argument(
        "--source-hdf5",
        type=Path,
        default=Path("data/mimicgen_data/source/threading.hdf5"),
    )
    parser.add_argument(
        "--output-hdf5",
        type=Path,
        default=Path("data/spine_cito/threading/refine_200_tune_c.hdf5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/spine_cito/threading"),
    )
    parser.add_argument("--delete-parts", action="store_true")
    args = parser.parse_args()

    python_bin = Path(".venv/bin/python")
    runner = Path("scripts/run_phase2_cito_batch.py")
    merger = Path("scripts/merge_hdf5_demos.py")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total = _count_demos(args.output_hdf5)
    demo_ids = _load_demo_ids(args.output_hdf5)

    current = args.start
    while total < args.target and current <= args.max_start:
        chunk_end = min(args.max_start + 1, current + args.chunk_size)
        success_parts: list[Path] = []

        for start in range(current, chunk_end):
            demo_name = f"demo_{start}"
            if demo_name in demo_ids:
                continue
            part_path = output_dir / f"refine_200_tune_c_part_{start}.hdf5"
            if part_path.exists():
                part_path.unlink()
            cmd = [
                str(python_bin),
                str(runner),
                "--input-hdf5",
                str(args.input_hdf5),
                "--output-hdf5",
                str(part_path),
                "--source-hdf5",
                str(args.source_hdf5),
                "--limit",
                "1",
                "--start",
                str(start),
                "--allow-fail",
                "--demo-timeout",
                str(args.demo_timeout),
                "--scvx-iters",
                str(args.scvx_iters),
                "--penalty-loops",
                str(args.penalty_loops),
                "--trust-region",
                str(args.trust_region),
                "--overwrite",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                if part_path.exists():
                    part_path.unlink()
                continue
            if _part_has_demo(part_path):
                success_parts.append(part_path)
            else:
                if part_path.exists():
                    part_path.unlink()

        if success_parts:
            merge_cmd = [
                str(python_bin),
                str(merger),
                "--output-hdf5",
                str(args.output_hdf5),
                "--inputs",
                *[str(p) for p in success_parts],
            ]
            subprocess.run(merge_cmd, check=True)
            if args.delete_parts:
                for p in success_parts:
                    p.unlink(missing_ok=True)

        total = _count_demos(args.output_hdf5)
        demo_ids = _load_demo_ids(args.output_hdf5)
        print(
            f"chunk {current}-{chunk_end - 1}: +{len(success_parts)} successes, total={total}"
        )
        current = chunk_end

    print(f"Done. total_success={total}, last_start={current - 1}")


if __name__ == "__main__":
    main()
