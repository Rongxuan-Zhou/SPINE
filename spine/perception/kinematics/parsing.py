"""Utility functions to parse external kinematic exports into SPINE data structures."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterable, List, Mapping, MutableSequence, Sequence

from .data_structures import KinematicFrame, KinematicTrajectory, TrajectoryMetadata

logger = logging.getLogger(__name__)


def _read_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _as_floats(values: Iterable[object]) -> List[float]:
    floats: List[float] = []
    for val in values:
        try:
            floats.append(float(val))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"无法将 {val!r} 转换为浮点数") from exc
    return floats


def _extract_first(mapping: Mapping[str, object], keys: Sequence[str], default: float | None = None) -> float:
    for key in keys:
        if key in mapping:
            return float(mapping[key])  # type: ignore[arg-type]
    if default is None:
        raise KeyError(f"未找到时间戳键 {keys}")
    return default


def _extract_pose(mapping: Mapping[str, object]) -> List[float]:
    for key in ("end_effector_pose", "ee_pose", "pose", "world_pose"):
        if key in mapping:
            pose = mapping[key]
            return _as_floats(pose if isinstance(pose, Sequence) else [])
    raise KeyError("未找到末端位姿字段 (end_effector_pose / ee_pose / pose / world_pose)")


def _extract_joint_positions(mapping: Mapping[str, object]) -> List[float]:
    for key in ("joint_positions", "joints", "joint_state"):
        if key in mapping:
            joints = mapping[key]
            return _as_floats(joints if isinstance(joints, Sequence) else [])
    return []


def parse_trajectory_json(path: Path, metadata: TrajectoryMetadata) -> KinematicTrajectory:
    """Parses a generic JSON trajectory with frames list."""
    payload = _read_json(path)
    frames_payload = payload.get("frames") or payload.get("trajectory")
    if not isinstance(frames_payload, Sequence):
        raise ValueError(f"{path} 不包含 frames 列表")
    frames: List[KinematicFrame] = []
    for idx, frame_entry in enumerate(frames_payload):
        if not isinstance(frame_entry, Mapping):
            raise ValueError(f"{path} 的第 {idx} 个帧不是映射类型")
        timestamp = _extract_first(frame_entry, ("timestamp", "time", "t"))
        ee_pose = _extract_pose(frame_entry)
        joints = _extract_joint_positions(frame_entry)
        frames.append(KinematicFrame(timestamp=timestamp, joint_positions=joints, end_effector_pose=ee_pose))
    return KinematicTrajectory(frames=frames, metadata=metadata)


def discover_json_trajectories(root: Path, clip_filter: Sequence[str] | None = None) -> List[Path]:
    """Find candidate JSON trajectory files under a root, applying optional clip filters."""
    all_json = sorted(root.rglob("*.json"))
    if not clip_filter:
        return all_json
    filtered: List[Path] = []
    for path in all_json:
        if any(token in path.name for token in clip_filter):
            filtered.append(path)
    return filtered


def parse_dexcap_episode_frames(
    episode_dir: Path,
    metadata: TrajectoryMetadata,
    pose_file: str = "pose_2.txt",
    joint_file: str = "right_hand_joint.txt",
    frame_dt: float = 0.033,
) -> KinematicTrajectory:
    """Parse raw DexCap frame_* folders into a trajectory."""
    frame_dirs = sorted(p for p in episode_dir.iterdir() if p.is_dir() and re.match(r"frame_\d+", p.name))
    frames: List[KinematicFrame] = []
    if not frame_dirs:
        raise FileNotFoundError(f"{episode_dir} 下未找到 frame_* 目录")
    for idx, frame_dir in enumerate(frame_dirs):
        pose_path = frame_dir / pose_file
        if not pose_path.exists():
            logger.debug("跳过帧 %s, 未找到 %s", frame_dir, pose_file)
            continue
        pose_values = _as_floats(pose_path.read_text(encoding="utf-8").split())
        if len(pose_values) < 7:
            raise ValueError(f"{pose_path} 末端位姿长度不足 7")
        ee_pose = pose_values[:7]
        joint_positions: MutableSequence[float] = []
        if joint_file:
            joint_path = frame_dir / joint_file
            if joint_path.exists():
                joint_positions = _as_floats(joint_path.read_text(encoding="utf-8").split())
        timestamp = idx * frame_dt
        frames.append(KinematicFrame(timestamp=timestamp, joint_positions=list(joint_positions), end_effector_pose=ee_pose))
    if not frames:
        raise FileNotFoundError(f"{episode_dir} 下未解析到任何帧")
    return KinematicTrajectory(frames=frames, metadata=metadata)


__all__ = [
    "discover_json_trajectories",
    "parse_dexcap_episode_frames",
    "parse_trajectory_json",
]
