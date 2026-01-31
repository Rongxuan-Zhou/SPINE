"""Thin MimicGen adapter wrapper with dependency detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Iterator, List

import h5py

from .augmentations import maybe_augment
from .configs import MimicGenConfig
from .data_structures import KinematicFrame, KinematicTrajectory, TrajectoryMetadata
from .dependencies import require_dependency
from .parsing import discover_json_trajectories, parse_trajectory_json
from .sources import KinematicSourceAdapter

logger = logging.getLogger(__name__)


class MimicGenAdapter(KinematicSourceAdapter):
    """Adapter for MimicGen-generated kinematic demonstrations."""

    def __init__(self, config: MimicGenConfig) -> None:
        super().__init__("mimicgen")
        self.config = config

    def _ensure_available(self) -> None:
        require_dependency(
            module_name="mimicgen",
            repo_name="MimicGen",
            repo_url="https://github.com/NVlabs/mimicgen",
            install_subdir="mimicgen",
        )

    def generate(self) -> Iterable[KinematicTrajectory]:
        self._ensure_available()
        json_paths = discover_json_trajectories(self.config.dataset_root)
        hdf5_paths = sorted(Path(self.config.dataset_root).glob("*.hdf5"))
        if not json_paths and not hdf5_paths:
            raise FileNotFoundError(
                f"{self.config.dataset_root} 下未发现 MimicGen 轨迹（JSON 或 HDF5）"
            )
        for path in json_paths:
            metadata = TrajectoryMetadata(
                source="mimicgen",
                clip_id=path.stem,
                augmentations=self.config.augmentations.as_dict(),
            )
            logger.debug("解析 MimicGen JSON 轨迹 %s", path)
            yield from maybe_augment(
                parse_trajectory_json(path, metadata),
                self.config.augmentations,
                num_aug=1,
            )
        for h5_path in hdf5_paths:
            logger.debug("解析 MimicGen HDF5 %s", h5_path)
            yield from self._load_hdf5(h5_path)

    def _load_hdf5(self, path: Path) -> Iterator[KinematicTrajectory]:
        with h5py.File(path, "r") as f:
            demos = list(f.get("data", {}).keys()) if "data" in f else []
            if not demos:
                logger.warning("HDF5 %s 未找到 data/* 演示，跳过", path)
                return
            for demo in demos:
                frames = self._extract_demo_frames(f, demo)
                if not frames:
                    logger.warning("HDF5 %s demo %s 为空，跳过", path, demo)
                    continue
                metadata = TrajectoryMetadata(
                    source="mimicgen",
                    clip_id=f"{path.stem}_{demo}",
                    augmentations=self.config.augmentations.as_dict(),
                )
                base_traj = KinematicTrajectory(frames=frames, metadata=metadata)
                yield from maybe_augment(
                    base_traj,
                    self.config.augmentations,
                    num_aug=self.config.max_augmentations_per_demo,
                )

    def _extract_demo_frames(self, f: h5py.File, demo: str) -> List[KinematicFrame]:
        pos_key = f"data/{demo}/obs/robot0_eef_pos"
        quat_key = f"data/{demo}/obs/robot0_eef_quat"
        joint_key = f"data/{demo}/obs/robot0_joint_pos"
        if pos_key not in f:
            return []
        pos = f[pos_key][:]
        quat = f[quat_key][:] if quat_key in f else None
        joints = f[joint_key][:] if joint_key in f else None
        frames: List[KinematicFrame] = []
        for idx, p in enumerate(pos):
            ee_pose = list(p.tolist())
            if quat is not None and idx < len(quat):
                ee_pose.extend(quat[idx].tolist())
            else:
                ee_pose.extend([0.0, 0.0, 0.0, 1.0])
            joint_positions = (
                joints[idx].tolist() if joints is not None and idx < len(joints) else []
            )
            frames.append(
                KinematicFrame(
                    timestamp=idx * 0.05,  # MimicGen 控制频率约 20Hz
                    joint_positions=joint_positions,
                    end_effector_pose=ee_pose,
                )
            )
        return frames


__all__ = ["MimicGenAdapter"]
