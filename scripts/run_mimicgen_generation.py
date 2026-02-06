#!/usr/bin/env python
"""Run MimicGen data generation with a local config and proper PYTHONPATH."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--robosuite-path", type=Path, default=None)
    parser.add_argument("--auto-remove-exp", action="store_true")
    parser.add_argument("--export-raw", action="store_true")
    parser.add_argument("--raw-q-key", type=str, default=None)
    parser.add_argument("--raw-rgb-key", type=str, default=None)
    parser.add_argument("--raw-force-key", type=str, default=None)
    parser.add_argument("--raw-success-name", type=str, default=None)
    parser.add_argument("--raw-failed-name", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    mimicgen_root = repo_root / "external" / "mimicgen"
    robomimic_stub = repo_root / "tools" / "robomimic_stub"
    script_path = mimicgen_root / "mimicgen" / "scripts" / "generate_dataset.py"

    if not script_path.exists():
        raise FileNotFoundError(f"Missing MimicGen script: {script_path}")

    env = os.environ.copy()
    robosuite_path = args.robosuite_path or env.get("ROBOSUITE_PATH")
    extra = ""
    if robosuite_path:
        extra = f"{robosuite_path}:"
    env["PYTHONPATH"] = f"{robomimic_stub}:{mimicgen_root}:{extra}{env.get('PYTHONPATH','')}"
    env.setdefault("MUJOCO_GL", "osmesa")
    env.setdefault("PYOPENGL_PLATFORM", "osmesa")

    cmd = [sys.executable, str(script_path), "--config", str(args.config)]
    if args.debug:
        cmd.append("--debug")
    if args.auto_remove_exp:
        cmd.append("--auto-remove-exp")
    if args.render:
        cmd.append("--render")
    if args.video:
        cmd.extend(["--video_path", str(args.video)])
    if args.export_raw:
        cmd.append("--export_raw")
    if args.raw_q_key:
        cmd.extend(["--raw_q_key", args.raw_q_key])
    if args.raw_rgb_key:
        cmd.extend(["--raw_rgb_key", args.raw_rgb_key])
    if args.raw_force_key:
        cmd.extend(["--raw_force_key", args.raw_force_key])
    if args.raw_success_name:
        cmd.extend(["--raw_success_name", args.raw_success_name])
    if args.raw_failed_name:
        cmd.extend(["--raw_failed_name", args.raw_failed_name])

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
