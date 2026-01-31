#!/usr/bin/env python
"""Recompute end-effector poses from joint trajectories using robosuite forward kinematics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import robosuite as suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute EE pose from joint_positions via robosuite (Panda).")
    parser.add_argument("input", type=Path, help="Input trajectory JSON with joint_positions")
    parser.add_argument("output", type=Path, help="Output JSON with updated end_effector_pose")
    parser.add_argument("--env", type=str, default="Lift", help="robosuite env name (default: Lift)")
    return parser.parse_args()


def load_frames(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    payload = load_frames(args.input)
    frames = payload.get("frames", [])
    if not frames:
        raise ValueError("输入 JSON 不含 frames")

    env = suite.make(
        args.env,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
    )
    robot = env.robots[0]

    updated_frames: List[Dict[str, Any]] = []
    for f in frames:
        joints = f.get("joint_positions", [])
        if len(joints) != len(robot._ref_joint_pos_indexes):
            raise ValueError(f"期望 {len(robot._ref_joint_pos_indexes)} 个关节，得到 {len(joints)}")
        q = np.array(joints, dtype=float)
        env.sim.data.qpos[robot._ref_joint_pos_indexes] = q
        env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0
        env.sim.forward()
        ee_pos = robot._hand_pos.astype(float).tolist()
        ee_quat = robot._hand_quat.astype(float).tolist()  # order as provided by robosuite (x,y,z,w)
        new_pose = ee_pos + ee_quat
        f_new = dict(f)
        f_new["end_effector_pose"] = new_pose
        updated_frames.append(f_new)

    payload["frames"] = updated_frames
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    print(f"Updated EE poses saved to {args.output}")


if __name__ == "__main__":
    main()
