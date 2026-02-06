#!/usr/bin/env python
"""Run N=50 training sweep for noforce/simforce/spine_refine groups."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase2 training sweep.")
    parser.add_argument("--trainsets-dir", type=Path, required=True)
    parser.add_argument("--tag", type=str, default="n50")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--force-dim", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--use-rgb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable RGB conditioning in DiT training (default: true).",
    )
    parser.add_argument("--rgb-key", type=str, default="agentview_rgb")
    parser.add_argument("--rgb-size", type=int, default=84)
    parser.add_argument(
        "--use-physics-inpainting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable masked physics token inpainting training (default: true).",
    )
    parser.add_argument("--physics-token-dim", type=int, default=3)
    parser.add_argument("--physics-mask-prob", type=float, default=0.5)
    parser.add_argument("--loss-phys-weight", type=float, default=0.5)
    parser.add_argument("--contact-force-threshold", type=float, default=2.0)
    parser.add_argument("--force-mag-clip", type=float, default=50.0)
    parser.add_argument(
        "--ckpt-root",
        type=Path,
        default=Path("data/checkpoints_threading_rgb_inpaint"),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have an ep200 checkpoint.",
    )
    parser.add_argument(
        "--resume-latest",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from latest checkpoint in each ckpt dir (default: true).",
    )
    return parser.parse_args()


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Training command failed with code {result.returncode}")


def main() -> None:
    args = _parse_args()
    tag = args.tag
    trainsets = args.trainsets_dir
    groups = {
        "noforce": trainsets / f"{tag}_noforce.hdf5",
        "simforce": trainsets / f"{tag}_simforce.hdf5",
        "spine_refine": trainsets / f"{tag}_spine_refine.hdf5",
    }
    optional_full = trainsets / f"{tag}_spine_full.hdf5"
    if optional_full.exists():
        groups["spine_full"] = optional_full
    for group, dataset in groups.items():
        if not dataset.exists():
            raise FileNotFoundError(f"[{group}] dataset missing: {dataset}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    for group, dataset in groups.items():
        for seed in seeds:
            ckpt_dir = args.ckpt_root / f"{tag}_{group}" / f"seed{seed}"
            ep200 = ckpt_dir / "spine_dit_ep200.pth"
            if args.skip_existing and ep200.exists():
                print(f"[skip] {ep200}", flush=True)
                continue
            cmd = [
                "python",
                "train_dit_min.py",
                "--dataset",
                str(dataset),
                "--ckpt-dir",
                str(ckpt_dir),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--horizon",
                str(args.horizon),
                "--diffusion-steps",
                str(args.diffusion_steps),
                "--force-dim",
                str(args.force_dim),
                "--lr",
                str(args.lr),
                "--rgb-key",
                str(args.rgb_key),
                "--rgb-size",
                str(args.rgb_size),
                "--physics-token-dim",
                str(args.physics_token_dim),
                "--physics-mask-prob",
                str(args.physics_mask_prob),
                "--loss-phys-weight",
                str(args.loss_phys_weight),
                "--contact-force-threshold",
                str(args.contact_force_threshold),
                "--force-mag-clip",
                str(args.force_mag_clip),
                "--seed",
                str(seed),
            ]
            if args.use_rgb:
                cmd.append("--use-rgb")
            else:
                cmd.append("--no-use-rgb")
            if args.use_physics_inpainting:
                cmd.append("--use-physics-inpainting")
            else:
                cmd.append("--no-use-physics-inpainting")
            if args.resume_latest:
                cmd.append("--resume-latest")
            _run(cmd)


if __name__ == "__main__":
    main()
