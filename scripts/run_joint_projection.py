#!/usr/bin/env python
"""Clamp/smooth 7-DOF joint trajectories with velocity/acceleration limits."""

from __future__ import annotations

import argparse
from pathlib import Path

from spine.planning.cito.joint_projection import JointProjectionConfig, project_joint_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth joint trajectories with vel/acc limits and FR3 bounds.")
    parser.add_argument("--input", type=Path, required=True, help="Input trajectory JSON (7-DOF joints)")
    parser.add_argument("--output", type=Path, required=True, help="Output path for clamped JSON")
    parser.add_argument("--dt", type=float, default=0.05, help="Timestep in seconds")
    parser.add_argument("--vel-limit", type=float, default=2.0, help="Max joint velocity (rad/s)")
    parser.add_argument("--acc-limit", type=float, default=5.0, help="Max joint acceleration (rad/s^2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = JointProjectionConfig(vel_limit=args.vel_limit, acc_limit=args.acc_limit, dt=args.dt)
    out = project_joint_trajectory(args.input, args.output, cfg)
    print(f"Joint-projected trajectory saved to {out}")


if __name__ == "__main__":
    main()
