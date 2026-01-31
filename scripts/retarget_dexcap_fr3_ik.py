#!/usr/bin/env python
"""
Retarget DexCap right-hand poses (pose_3.txt 4x4 matrix) to FR3 joint angles via simple IK.

Usage:
  MUJOCO_GL=osmesa python scripts/retarget_dexcap_fr3_ik.py \\
      /data/dexcap/rawdata_wipe_1-14/save_wipe_1-14/save_data_wipe_1-14_01 \\
      --mjcf /home/rongxuan_zhou/mujoco_menagerie/franka_fr3/fr3.xml \\
      --output data/kinematics/dexcap/fr3_ik_save_data_wipe_1-14_01.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import mujoco
import numpy as np


def _load_pose_matrix(path: Path) -> np.ndarray:
    vals = np.fromstring(path.read_text(), sep=" ")
    if vals.size != 16:
        raise ValueError(f"{path} expected 16 floats (4x4), got {vals.size}")
    return vals.reshape(4, 4)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation to quaternion (w, x, y, z)."""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=float)


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Log map from SO(3) to axis-angle vector (length=angle)."""
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (
        2.0 * math.sin(theta)
    )
    return axis * theta


def ik_solve(
    model: mujoco.MjModel,
    siteid: int,
    target_pos: np.ndarray,
    target_rot: np.ndarray,
    init_q: np.ndarray,
    iters: int = 200,
    pos_tol: float = 1e-4,
    rot_tol: float = 1e-3,
    step_size: float = 0.7,
    damping: float = 1e-4,
) -> np.ndarray:
    """Damped least-squares IK to match site pose."""
    data = mujoco.MjData(model)
    q = init_q.copy()
    jrange = model.jnt_range[:7]
    for _ in range(iters):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[siteid].copy()
        rot = data.site_xmat[siteid].reshape(3, 3).copy()
        pos_err = target_pos - pos
        rot_err_vec = _so3_log(target_rot @ rot.T)
        err = np.concatenate([pos_err, rot_err_vec])
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err_vec) < rot_tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, siteid)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        H = J.T @ J + damping * np.eye(7)
        dq = np.linalg.solve(H, J.T @ err * step_size)
        q = np.clip(q + dq, jrange[:, 0], jrange[:, 1])
    return q


def retarget_episode(
    episode_dir: Path, mjcf: Path, output: Path, horizon: int | None = None
) -> None:
    model = mujoco.MjModel.from_xml_path(str(mjcf))
    siteid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    # Use keyframe "home" if present, else zeros.
    if model.nkey > 0:
        q_home = np.asarray(model.key_qpos).reshape(-1, model.nq)[0]
    else:
        q_home = np.zeros(model.nq)

    frame_dirs = sorted(
        [p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")]
    )
    if horizon:
        frame_dirs = frame_dirs[:horizon]
    if not frame_dirs:
        raise FileNotFoundError(f"No frame_* dirs under {episode_dir}")

    frames_out: List[dict] = []
    q_prev = q_home[:7].copy()
    dt = 0.033  # raw frame spacing ~30Hz from DexCap README
    prev_pos = None
    alpha_pos = 0.2
    z_min = 0.0
    for idx, fdir in enumerate(frame_dirs):
        pose_path = fdir / "pose_3.txt"
        if not pose_path.exists():
            continue
        T = _load_pose_matrix(pose_path)
        target_pos = T[:3, 3]
        if prev_pos is None:
            prev_pos = target_pos.copy()
        target_pos = alpha_pos * target_pos + (1 - alpha_pos) * prev_pos
        target_pos[2] = max(target_pos[2], z_min)
        prev_pos = target_pos.copy()
        target_rot = T[:3, :3]
        q_sol = ik_solve(model, siteid, target_pos, target_rot, q_prev)
        q_prev = q_sol
        target_quat = _rot_to_quat(target_rot)
        frames_out.append(
            {
                "timestamp": idx * dt,
                "joint_positions": q_sol.tolist(),
                "end_effector_pose": [*target_pos.tolist(), *target_quat.tolist()],
            }
        )

    payload = {
        "frames": frames_out,
        "source": "dexcap_fr3_ik",
        "clip": episode_dir.name,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    print(f"Saved {len(frames_out)} frames to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retarget DexCap pose_3 matrices to FR3 joints via IK."
    )
    parser.add_argument(
        "episode_dir",
        type=Path,
        help="DexCap episode directory containing frame_* folders",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        required=True,
        help="FR3 MJCF path (e.g., menagerie fr3.xml)",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output trajectory JSON path"
    )
    parser.add_argument(
        "--horizon", type=int, default=None, help="Optional frame cap for quick tests"
    )
    args = parser.parse_args()
    retarget_episode(args.episode_dir, args.mjcf, args.output, args.horizon)


if __name__ == "__main__":
    main()
