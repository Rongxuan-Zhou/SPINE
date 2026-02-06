#!/usr/bin/env python
"""Inspect MimicGen dataset for wrench keys and basic signal stats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--wrench-key", type=str, default="robot0_ee_wrench")
    parser.add_argument("--demo", type=str, default=None)
    parser.add_argument("--force-threshold", type=float, default=2.0)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        demos = list(f["data"].keys())
        if not demos:
            raise RuntimeError("No demos found in dataset.")
        demo = args.demo or demos[0]
        grp = f[f"data/{demo}"]
        obs = grp["obs"]
        keys = list(obs.keys())
        print(f"Dataset: {args.hdf5}")
        print(f"Total demos: {len(demos)} | Inspect demo: {demo}")
        print(f"Obs keys (sample): {sorted(keys)[:12]}")
        print(f"Has wrench key '{args.wrench_key}': {args.wrench_key in obs}")
        if "success" in grp.attrs:
            print(f"Demo success attr: {bool(grp.attrs['success'])}")
        else:
            print("Demo success attr: MISSING")

        if args.wrench_key not in obs:
            return
        wrench = np.array(obs[args.wrench_key], dtype=float)
        force = np.linalg.norm(wrench[:, :3], axis=1)
        diff = np.diff(force)
        hf_energy = float(np.mean(diff**2))
        contact_mask = force > args.force_threshold
        contact_ratio = float(np.mean(contact_mask))
        print(f"Wrench shape: {wrench.shape}")
        print(f"Force mean: {force.mean():.3f} | std: {force.std():.3f}")
        print(f"High-freq energy (diff^2 mean): {hf_energy:.6f}")
        print(f"Contact ratio (> {args.force_threshold}N): {contact_ratio:.3f}")


if __name__ == "__main__":
    main()
