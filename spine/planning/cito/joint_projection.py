"""Velocity/acceleration-constrained joint projection for 7-DOF Franka/Panda."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from spine.perception.kinematics.data_structures import KinematicFrame, KinematicTrajectory, TrajectoryMetadata
from spine.perception.kinematics.fr3_mapping import FR3_JOINT_NAMES


# Franka joint limits (rad), symmetric for Panda/FR3.
JOINT_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])


@dataclass
class JointProjectionConfig:
    vel_limit: float = 2.0  # rad/s
    acc_limit: float = 5.0  # rad/s^2
    dt: float = 0.05


def _load_trajectory(path: Path) -> KinematicTrajectory:
    payload = json.loads(path.read_text())
    meta = payload.get("metadata", {})
    frames = []
    for f in payload["frames"]:
        frames.append(
            KinematicFrame(
                timestamp=float(f["timestamp"]),
                joint_positions=f.get("joint_positions", []),
                end_effector_pose=f["end_effector_pose"],
                base_frame=f.get("base_frame", "world"),
            )
        )
    traj = KinematicTrajectory(frames=frames, metadata=TrajectoryMetadata(**meta))
    return traj


def _clamp_joint_limits(q: np.ndarray) -> np.ndarray:
    return np.clip(q, JOINT_LOWER, JOINT_UPPER)


def _smooth_joints(joints: np.ndarray, cfg: JointProjectionConfig) -> np.ndarray:
    """Clamp velocity/acceleration step-to-step for a joint trajectory."""
    out = joints.copy()
    dt = cfg.dt
    vel_max = cfg.vel_limit * dt
    acc_max = cfg.acc_limit * dt * dt
    for t in range(1, out.shape[0]):
        dq = out[t] - out[t - 1]
        low = JOINT_LOWER - out[t - 1]
        high = JOINT_UPPER - out[t - 1]
        range_min = np.maximum(-vel_max, low)
        range_max = np.minimum(vel_max, high)
        # ensure range_min <= range_max even when limits are tighter than vel bound
        range_min = np.minimum(range_min, range_max)
        range_max = np.maximum(range_min, range_max)
        dq = np.clip(dq, range_min, range_max)
        out[t] = out[t - 1] + dq
        if t >= 2:
            ddq = out[t] - 2 * out[t - 1] + out[t - 2]
            ddq = np.clip(ddq, -acc_max, acc_max)
            out[t] = out[t - 1] + dq - ddq
        out[t] = _clamp_joint_limits(out[t])
    return out


def project_joint_trajectory(
    input_path: Path,
    output_path: Path,
    cfg: JointProjectionConfig | None = None,
) -> Path:
    traj = _load_trajectory(input_path)
    cfg = cfg or JointProjectionConfig()
    # Require 7-DOF
    joints = [np.array(f.joint_positions, dtype=float) for f in traj.frames]
    if not joints or len(joints[0]) != len(FR3_JOINT_NAMES):
        raise ValueError("需要 7 维关节序列才能做关节空间投影")
    q = np.stack(joints, axis=0)
    q_proj = _smooth_joints(_clamp_joint_limits(q), cfg)

    new_frames: List[dict[str, object]] = []
    for frame, q_new in zip(traj.frames, q_proj):
        new_frames.append(
            {
                "timestamp": frame.timestamp,
                "joint_positions": list(q_new),
                "end_effector_pose": list(frame.end_effector_pose),
                "base_frame": frame.base_frame,
            }
        )
    meta = traj.metadata.as_dict()
    meta.setdefault("notes", {})  # type: ignore[assignment]
    meta["notes"]["joint_projection"] = "clamped vel/acc with FR3 limits"  # type: ignore[index]
    payload = {"metadata": meta, "frames": new_frames}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    return output_path


__all__ = ["project_joint_trajectory", "JointProjectionConfig"]
