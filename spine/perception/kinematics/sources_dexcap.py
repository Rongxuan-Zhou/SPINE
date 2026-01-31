"""Thin DexCap adapter wrapper with dependency detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional
import math
import numpy as np

from .configs import DexCapConfig
from .data_structures import KinematicFrame, KinematicTrajectory, TrajectoryMetadata
from .parsing import parse_dexcap_episode_frames
from .sources import KinematicSourceAdapter
from .augmentations import maybe_augment

logger = logging.getLogger(__name__)

# Placeholder FR3 arm joint indexes from DexCap sequence
# DexCap joint files contain 63 floats (21 markers * xyz). We currently slice
# the first 7 values as a stub to satisfy 7-DOF interfaces; replace with a real
# retargeting map when available.
FR3_JOINT_INDEXES = list(range(7))


class DexCapAdapter(KinematicSourceAdapter):
    """Adapter for DexCap dexterous manipulation clips."""

    def __init__(self, config: DexCapConfig) -> None:
        super().__init__("dexcap")
        self.config = config
        self.num_aug = 9  # how many augmented copies per clip

    def _ensure_available(self) -> None:
        # DexCap repo is data-only here; Python package not required. For IK we need mujoco.
        if self.config.use_ik:
            try:
                import mujoco  # noqa: F401
            except ImportError as exc:  # pragma: no cover - runtime guard
                raise ImportError(
                    "DexCap IK 需要 mujoco，请先 `pip install mujoco` (或安装本地 wheel) 后重试。"
                ) from exc

    def generate(self) -> Iterable[KinematicTrajectory]:
        self._ensure_available()
        root = self.config.dataset_root
        episode_dirs = _find_dexcap_episode_dirs(root, self.config.clip_filter)
        for episode_dir in episode_dirs:
            metadata = TrajectoryMetadata(source="dexcap", clip_id=episode_dir.name, augmentations={})
            logger.debug("解析 DexCap 原始帧目录 %s", episode_dir)
            if self.config.use_ik:
                if not self.config.mjcf_path:
                    raise ValueError("DexCap use_ik=True 但未提供 mjcf_path")
                traj = self._retarget_with_ik(episode_dir, metadata, self.config.mjcf_path, self.config.ik_horizon)
            else:
                traj = self._project_to_fr3(parse_dexcap_episode_frames(episode_dir, metadata))
            yield from maybe_augment(traj, self.config.augmentations, num_aug=self.num_aug)

    def _project_to_fr3(self, traj: KinematicTrajectory) -> KinematicTrajectory:
        frames = []
        warned = False
        for f in traj.frames:
            joints = list(f.joint_positions)
            if len(joints) < len(FR3_JOINT_INDEXES):
                continue
            fr3_joints = [joints[i] for i in FR3_JOINT_INDEXES]
            if not warned:
                logger.debug("DexCap->FR3 using placeholder slice of 7/len=%d joints", len(joints))
                warned = True
            frames.append(
                type(f)(
                    timestamp=f.timestamp,
                    joint_positions=fr3_joints,
                    end_effector_pose=f.end_effector_pose,
                    base_frame=f.base_frame,
                )
            )
        return type(traj)(frames=frames, metadata=traj.metadata)

    # ---------- IK retargeting path ----------
    def _retarget_with_ik(
        self, episode_dir: Path, metadata: TrajectoryMetadata, mjcf_path: Path, horizon: Optional[int]
    ) -> KinematicTrajectory:
        import mujoco
        import numpy as np

        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        siteid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
        if siteid < 0:
            raise ValueError("FR3 MJCF 中未找到 site: attachment_site")
        if model.nkey > 0:
            q_home = np.asarray(model.key_qpos).reshape(-1, model.nq)[0]
        else:
            q_home = np.zeros(model.nq)
        # Compute home site position for base alignment
        data = mujoco.MjData(model)
        data.qpos[: model.nq] = q_home
        mujoco.mj_forward(model, data)
        site_home = data.site_xpos[siteid].copy()
        frame_dirs = sorted([p for p in episode_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")])
        if horizon:
            frame_dirs = frame_dirs[:horizon]
        if not frame_dirs:
            raise FileNotFoundError(f"{episode_dir} 下未找到 frame_* 目录")

        frames: List[KinematicFrame] = []
        q_prev = q_home[:7].copy()
        dt = self.config.frame_dt
        warned = False
        z_min = 0.0  # clamp above table
        alpha_pos = 0.7
        alpha_rot = 0.5
        fix_normal = True
        follow_yaw = False  # set False to keep constant orientation

        # preload poses
        poses = []
        for fdir in frame_dirs:
            pose_path = fdir / "pose_3.txt"
            if pose_path.exists():
                poses.append(_load_pose_matrix(pose_path))
        if not poses:
            raise FileNotFoundError(f"{episode_dir} 下未找到 pose_3.txt")

        def smooth_seq(vals: np.ndarray, kernel: int = 21, poly: int = 3) -> np.ndarray:
            """Savitzky-Golay if available, else moving average."""
            if len(vals) < kernel:
                return vals
            try:
                from scipy.signal import savgol_filter  # type: ignore

                # axis=0 handles vector rows
                return savgol_filter(vals, kernel_length=kernel, polyorder=poly, axis=0, mode="nearest")
            except Exception:
                pad = kernel // 2
                padded = np.pad(vals, ((pad, pad), (0, 0)), mode="edge")
                out = []
                for i in range(len(vals)):
                    window = padded[i : i + kernel]
                    out.append(window.mean(axis=0))
                return np.array(out)

        target_positions = np.array([T[:3, 3] for T in poses])
        target_positions[:, 2] = np.maximum(target_positions[:, 2], z_min)
        target_positions = smooth_seq(target_positions, kernel=21, poly=3)
        # Align targets to FR3 base so that initial site matches home pose
        if len(target_positions) > 0:
            offset = target_positions[0] - site_home
            target_positions = target_positions - offset

        yaws = np.array([math.atan2(T[1, 0], T[0, 0]) for T in poses])
        yaws = smooth_seq(yaws.reshape(-1, 1), kernel=21, poly=3).flatten()
        # if not following yaw, fix to initial yaw
        if not follow_yaw and len(yaws) > 0:
            yaws[:] = yaws[0]

        prev_pos: Optional[np.ndarray] = None
        prev_rot: Optional[np.ndarray] = None
        for idx, T in enumerate(poses):
            target_pos = target_positions[idx]
            if prev_pos is None:
                prev_pos = target_pos.copy()
            target_pos = alpha_pos * target_pos + (1 - alpha_pos) * prev_pos
            target_pos[2] = max(target_pos[2], z_min)
            prev_pos = target_pos.copy()

            yaw = yaws[idx] if (fix_normal and follow_yaw) else (yaws[0] if len(yaws) else 0.0)
            if not follow_yaw:
                yaw = 0.0  # keep constant orientation to reduce drift
            cy, sy = math.cos(yaw), math.sin(yaw)
            target_rot = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
            if prev_rot is None:
                prev_rot = target_rot.copy()
            target_rot = _slerp_so3(prev_rot, target_rot, alpha_rot)
            prev_rot = target_rot.copy()

            q_sol = _ik_solve(model, siteid, target_pos, target_rot, q_prev)
            q_prev = q_sol
            quat = _rot_to_quat(target_rot)
            frames.append(
                KinematicFrame(
                    timestamp=idx * dt,
                    joint_positions=q_sol.tolist(),
                    end_effector_pose=[*target_pos.tolist(), *list(quat)],
                    base_frame=metadata.clip_id,
                )
            )
            if not warned:
                logger.debug("DexCap IK retargeting启用，首帧 q=%s", q_sol)
                warned = True
        return KinematicTrajectory(frames=frames, metadata=metadata)


def _load_pose_matrix(path: Path):
    import numpy as np
    vals = np.fromstring(path.read_text(), sep=" ")
    if vals.size != 16:
        raise ValueError(f"{path} expected 16 floats (4x4), got {vals.size}")
    return vals.reshape(4, 4)


def _rot_to_quat(R):
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
    return [w, x, y, z]


def _so3_log(R):
    import numpy as np
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = math.acos(cos_theta)
    if theta < 1e-6:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(theta))
    return axis * theta


def _ik_solve(model, siteid, target_pos, target_rot, init_q, iters=200, pos_tol=1e-4, rot_tol=1e-3, step_size=0.15, damping=1e-4):
    import numpy as np
    import mujoco
    data = mujoco.MjData(model)
    q = init_q.copy()
    jrange = model.jnt_range[:7]
    dq_max = 0.008  # rad per step to limit spikes
    alpha_q = 0.4  # smooth joint update
    for _ in range(iters):
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[siteid].copy()
        rot = data.site_xmat[siteid].reshape(3, 3).copy()
        pos_err = target_pos - pos
        rot_err_vec = _so3_log(target_rot @ rot.T)
        if np.linalg.norm(pos_err) < pos_tol and np.linalg.norm(rot_err_vec) < rot_tol:
            break
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jacp, jacr, siteid)
        J = np.vstack([jacp[:, :7], jacr[:, :7]])
        H = J.T @ J + damping * np.eye(7)
        dq = np.linalg.solve(H, J.T @ np.concatenate([pos_err, rot_err_vec]) * step_size)
        dq = np.clip(dq, -dq_max, dq_max)
        q_candidate = q + dq
        q = alpha_q * q_candidate + (1 - alpha_q) * q
        q = np.clip(q, jrange[:, 0], jrange[:, 1])
    return q


def _slerp_so3(R0, R1, alpha):
    """Interpolate two rotation matrices via log/exp."""
    import numpy as np
    log_delta = _so3_log(R1 @ R0.T)
    delta = np.linalg.norm(log_delta)
    if delta < 1e-8:
        return R1
    axis = log_delta / delta
    theta = alpha * delta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R_delta = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)
    return R_delta @ R0


def _find_dexcap_episode_dirs(root: Path, clip_filter: Iterable[str]) -> list[Path]:
    episodes = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        if clip_filter and not any(token in path.name for token in clip_filter):
            continue
        if any(child.name.startswith("frame_") for child in path.iterdir() if child.is_dir()):
            episodes.append(path)
    return sorted(episodes)


__all__ = ["DexCapAdapter"]
