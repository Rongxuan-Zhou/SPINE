"""Lightweight augmentation utilities for kinematic trajectories."""

from __future__ import annotations

import math
import random
from typing import Iterable, List

import numpy as np

from .configs import AugmentationConfig
from .data_structures import KinematicFrame, KinematicTrajectory, TrajectoryMetadata


def _perturb_quaternion(q: List[float], rotation_noise_deg: float) -> List[float]:
    if rotation_noise_deg <= 0:
        return q
    # Sample a small axis-angle jitter and compose with original quaternion.
    axis = np.random.normal(size=3)
    norm = np.linalg.norm(axis)
    if norm < 1e-8:
        return q
    axis = axis / norm
    angle_rad = math.radians(rotation_noise_deg) * np.random.uniform(-1.0, 1.0)
    half = angle_rad / 2.0
    dq = np.array([axis[0] * math.sin(half), axis[1] * math.sin(half), axis[2] * math.sin(half), math.cos(half)])
    q_orig = np.array(q)
    # Quaternion multiplication dq * q_orig
    x1, y1, z1, w1 = dq
    x2, y2, z2, w2 = q_orig
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    q_new = np.array([x, y, z, w])
    q_new = q_new / np.linalg.norm(q_new)
    return q_new.tolist()


def _apply_noise(frame: KinematicFrame, cfg: AugmentationConfig) -> KinematicFrame:
    ee = list(frame.end_effector_pose)
    pos = np.array(ee[:3], dtype=float)
    quat = ee[3:]
    pos_noise = np.random.normal(scale=cfg.position_noise_m, size=3)
    pos_noisy = (pos + pos_noise).tolist()
    quat_noisy = _perturb_quaternion(quat, cfg.rotation_noise_deg)
    return KinematicFrame(
        timestamp=frame.timestamp,
        joint_positions=list(frame.joint_positions),
        end_effector_pose=pos_noisy + quat_noisy,
        base_frame=frame.base_frame,
    )


def apply_augmentations(
    trajectory: KinematicTrajectory, cfg: AugmentationConfig, time_warp: bool = True
) -> KinematicTrajectory:
    """Return a lightly augmented copy of the trajectory."""
    frames: List[KinematicFrame] = []
    warp_scale = 1.0
    if time_warp and cfg.time_warp_factor > 0:
        warp_scale = random.uniform(1.0 - cfg.time_warp_factor, 1.0 + cfg.time_warp_factor)
    for frame in trajectory.frames:
        augmented = _apply_noise(frame, cfg)
        augmented.timestamp = frame.timestamp * warp_scale
        frames.append(augmented)
    metadata = TrajectoryMetadata(
        source=trajectory.metadata.source,
        clip_id=trajectory.metadata.clip_id + "_aug",
        augmentations=cfg.as_dict(),
        notes=dict(trajectory.metadata.notes),
    )
    return KinematicTrajectory(frames=frames, metadata=metadata)


def maybe_augment(
    trajectory: KinematicTrajectory, cfg: AugmentationConfig, num_aug: int = 1
) -> Iterable[KinematicTrajectory]:
    """Yield the original trajectory and a small number of augmented variants."""
    yield trajectory
    for _ in range(max(0, num_aug)):
        yield apply_augmentations(trajectory, cfg)


__all__ = ["apply_augmentations", "maybe_augment"]
