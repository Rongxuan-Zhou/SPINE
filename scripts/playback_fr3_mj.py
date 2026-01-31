#!/usr/bin/env python
"""Minimal MuJoCo playback for FR3 MJCF trajectories (7-DOF joints)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import mujoco
import mujoco.viewer  # register viewer submodule
import numpy as np


def load_joints(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    joints = [f["joint_positions"] for f in data["frames"]]
    arr = np.array(joints, dtype=float)
    if arr.shape[1] != 7:
        raise ValueError(f"Expected 7 joints, got {arr.shape}")
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play FR3 joint trajectory using MuJoCo."
    )
    parser.add_argument(
        "trajectory", type=Path, help="Trajectory JSON with 7-DOF joints"
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        required=True,
        help="Path to FR3 MJCF (e.g., menagerie fr3.xml)",
    )
    parser.add_argument("--render", action="store_true", help="Enable GLFW viewer")
    parser.add_argument(
        "--horizon", type=int, default=None, help="Optional truncation length"
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1 / 60.0,
        help="Playback timestep (s), default 60 Hz",
    )
    parser.add_argument(
        "--playback-rate",
        type=float,
        default=1.0,
        help="Playback speed multiplier (1.0 = real-time at dt)",
    )
    args = parser.parse_args()

    q_traj = load_joints(args.trajectory)
    if args.horizon:
        q_traj = q_traj[: args.horizon]

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)

    viewer = mujoco.viewer.launch_passive(model, data) if args.render else None
    import time

    step_dt = args.dt / max(args.playback_rate, 1e-6)

    for q in q_traj:
        data.qpos[:7] = q
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
            if step_dt > 0:
                time.sleep(step_dt)
    if viewer:
        viewer.close()
    print(f"Replayed {len(q_traj)} steps with MJCF {args.mjcf}")


if __name__ == "__main__":
    main()
