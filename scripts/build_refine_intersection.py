#!/usr/bin/env python
"""Build intersection manifest for Refine datasets (Sim-Force vs CITO-Force).

This script identifies demos that exist in both the raw MimicGen dataset and the
CITO refine output, validates length alignment, and produces a manifest for
N-sweep sampling with fixed seeds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np


@dataclass
class SplitRatios:
    train: float
    val: float
    test: float

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not np.isclose(total, 1.0):
            raise ValueError(f"split ratios must sum to 1.0, got {total:.3f}")


def _parse_ratios(raw: str) -> SplitRatios:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Expected --split-ratios like '0.8,0.1,0.1'")
    ratios = SplitRatios(*(float(p) for p in parts))
    ratios.validate()
    return ratios


def _sorted_demo_keys(keys: List[str]) -> List[str]:
    def sort_key(name: str):
        if name.startswith("demo_"):
            suffix = name.split("_", 1)[-1]
            if suffix.isdigit():
                return int(suffix)
        return name

    return sorted(keys, key=sort_key)


def _demo_length_raw(demo_grp: h5py.Group, joint_key: str) -> int:
    obs = demo_grp["obs"]
    return int(obs[joint_key].shape[0])


def _demo_length_cito(demo_grp: h5py.Group) -> int:
    return int(demo_grp["q_ref"].shape[0])


def _build_splits(
    rng: np.random.Generator, demos: List[str], ratios: SplitRatios
) -> Dict[str, List[str]]:
    shuffled = demos.copy()
    rng.shuffle(shuffled)
    total = len(shuffled)
    n_train = int(total * ratios.train)
    n_val = int(total * ratios.val)
    n_test = total - n_train - n_val
    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }
    return splits


def _build_n_sweep(
    train_demos: List[str], n_values: List[int]
) -> Dict[str, List[str]]:
    sweep: Dict[str, List[str]] = {}
    for n in n_values:
        if n > len(train_demos):
            raise ValueError(
                f"N={n} exceeds train split size {len(train_demos)}; "
                "increase raw count or reduce N."
            )
        sweep[str(n)] = train_demos[:n]
    return sweep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build intersection manifest for Refine datasets."
    )
    parser.add_argument("--raw-hdf5", type=Path, required=True)
    parser.add_argument("--cito-hdf5", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument(
        "--n-sweep",
        type=str,
        default="50,100,200,500,1000",
        help="Comma-separated N values for sweep (default: 50,100,200,500,1000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--joint-key", type=str, default="robot0_joint_pos")
    parser.add_argument("--wrench-key", type=str, default="robot0_ee_wrench")
    parser.add_argument(
        "--allow-failed",
        action="store_true",
        default=False,
        help="Include demos without success attr (default: require success).",
    )
    parser.add_argument(
        "--split-ratios",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test ratios (default: 0.8,0.1,0.1).",
    )
    args = parser.parse_args()

    n_values = [int(x.strip()) for x in args.n_sweep.split(",") if x.strip()]
    if not n_values:
        raise ValueError("No N values provided; check --n-sweep input.")
    ratios = _parse_ratios(args.split_ratios)

    skipped_missing_wrench: List[str] = []
    skipped_missing_joint: List[str] = []
    skipped_length_mismatch: List[Tuple[str, int, int]] = []
    skipped_not_success: List[str] = []

    with h5py.File(args.raw_hdf5, "r") as f_raw, h5py.File(
        args.cito_hdf5, "r"
    ) as f_cito:
        raw_demos = set(f_raw["data"].keys())
        cito_demos = set(f_cito["data"].keys())
        intersection = _sorted_demo_keys(list(raw_demos & cito_demos))

        eligible: List[str] = []
        demo_lengths: Dict[str, int] = {}

        for demo in intersection:
            demo_raw = f_raw["data"][demo]
            demo_cito = f_cito["data"][demo]

            if "obs" not in demo_raw or args.joint_key not in demo_raw["obs"]:
                skipped_missing_joint.append(demo)
                continue
            if args.wrench_key not in demo_raw["obs"]:
                skipped_missing_wrench.append(demo)
                continue
            if (not args.allow_failed) and not bool(demo_raw.attrs.get("success", 0)):
                skipped_not_success.append(demo)
                continue
            raw_len = _demo_length_raw(demo_raw, args.joint_key)
            cito_len = _demo_length_cito(demo_cito)
            if raw_len != cito_len:
                skipped_length_mismatch.append((demo, raw_len, cito_len))
                continue
            eligible.append(demo)
            demo_lengths[demo] = raw_len

    rng = np.random.default_rng(args.seed)
    splits = _build_splits(rng, eligible, ratios)
    n_sweep = _build_n_sweep(splits["train"], n_values)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_hdf5": str(args.raw_hdf5),
        "cito_hdf5": str(args.cito_hdf5),
        "seed": args.seed,
        "joint_key": args.joint_key,
        "wrench_key": args.wrench_key,
        "require_success": bool(not args.allow_failed),
        "split_ratios": asdict(ratios),
        "counts": {
            "raw_total": len(raw_demos),
            "cito_total": len(cito_demos),
            "intersection": len(intersection),
            "eligible": len(eligible),
        },
        "skipped": {
            "missing_joint": skipped_missing_joint,
            "missing_wrench": skipped_missing_wrench,
            "not_success": skipped_not_success,
            "length_mismatch": [
                {"demo": demo, "raw_len": raw_len, "cito_len": cito_len}
                for demo, raw_len, cito_len in skipped_length_mismatch
            ],
        },
        "demo_lengths": demo_lengths,
        "splits": splits,
        "n_sweep": n_sweep,
    }

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    print("✅ Intersection manifest written")
    print(f"  path: {args.out_manifest}")
    print(
        "  counts:",
        f"raw={manifest['counts']['raw_total']},",
        f"cito={manifest['counts']['cito_total']},",
        f"intersection={manifest['counts']['intersection']},",
        f"eligible={manifest['counts']['eligible']}",
    )
    print("  n_sweep:", ", ".join(f"{k}:{len(v)}" for k, v in n_sweep.items()))


if __name__ == "__main__":
    main()
