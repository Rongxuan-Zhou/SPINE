#!/usr/bin/env python3
"""Read-only tqdm monitor for Phase2 training checkpoints.

This script only scans checkpoint directories and prints progress bars.
It does not open model files for writing and will not interfere with training.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required. Install with: pip install tqdm"
    ) from exc


GROUPS = ("noforce", "simforce", "spine_refine")
EP_RE = re.compile(r"spine_dit_ep(\d+)\.pth$")


@dataclass(frozen=True)
class Job:
    group: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.group}/seed{self.seed}"


def _latest_epoch(seed_dir: Path) -> int:
    if not seed_dir.exists():
        return 0
    best = 0
    for p in seed_dir.glob("spine_dit_ep*.pth"):
        m = EP_RE.match(p.name)
        if not m:
            continue
        best = max(best, int(m.group(1)))
    return best


def _collect_epochs(
    ckpt_root: Path,
    tag: str,
    seeds: List[int],
    target_epoch: int,
) -> Dict[Job, int]:
    out: Dict[Job, int] = {}
    for group in GROUPS:
        for seed in seeds:
            job = Job(group=group, seed=seed)
            d = ckpt_root / f"{tag}_{group}" / f"seed{seed}"
            out[job] = min(_latest_epoch(d), target_epoch)
    return out


def _format_eta(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--:--:--"
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="tqdm monitor for Phase2 training progress")
    parser.add_argument("--ckpt-root", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="n1000")
    parser.add_argument("--target-epoch", type=int, default=200)
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--refresh-sec", type=float, default=5.0)
    parser.add_argument(
        "--stop-when-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit automatically when all jobs reach target epoch.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit (useful for testing).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    jobs = [Job(group=g, seed=s) for g in GROUPS for s in seeds]

    total_epochs = len(jobs) * args.target_epoch
    summary = tqdm(total=total_epochs, desc="overall", position=0, dynamic_ncols=True)

    bars: Dict[Job, tqdm] = {}
    for idx, job in enumerate(jobs, start=1):
        bars[job] = tqdm(
            total=args.target_epoch,
            desc=job.key,
            position=idx,
            leave=True,
            dynamic_ncols=True,
        )

    last_total = 0
    last_time = time.time()

    try:
        while True:
            epochs = _collect_epochs(
                ckpt_root=args.ckpt_root,
                tag=args.tag,
                seeds=seeds,
                target_epoch=args.target_epoch,
            )

            done_jobs = 0
            total_done = 0
            for job in jobs:
                ep = epochs[job]
                b = bars[job]
                b.n = ep
                b.refresh()
                total_done += ep
                if ep >= args.target_epoch:
                    done_jobs += 1

            now = time.time()
            dt = max(1e-6, now - last_time)
            dprog = total_done - last_total
            speed = dprog / dt  # epochs/sec over recent interval
            eta = None
            if speed > 0:
                eta = (total_epochs - total_done) / speed

            summary.n = total_done
            summary.set_postfix(
                done=f"{done_jobs}/{len(jobs)}",
                eta=_format_eta(eta),
                rate=f"{speed:.2f} ep/s",
            )
            summary.refresh()

            last_total = total_done
            last_time = now

            if args.once:
                break
            if args.stop_when_done and done_jobs == len(jobs):
                break
            time.sleep(max(0.2, args.refresh_sec))
    except KeyboardInterrupt:
        tqdm.write("Stopped by user.")
    finally:
        for b in bars.values():
            b.close()
        summary.close()


if __name__ == "__main__":
    main()
