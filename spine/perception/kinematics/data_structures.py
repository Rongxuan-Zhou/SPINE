"""Lightweight data structures for kinematic trajectories."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


@dataclass
class TrajectoryMetadata:
    """Metadata describing the origin of a kinematic trajectory."""

    source: str
    clip_id: str
    augmentations: Dict[str, float] = field(default_factory=dict)
    notes: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "clip_id": self.clip_id,
            "augmentations": dict(self.augmentations),
            "notes": dict(self.notes),
        }


@dataclass
class KinematicFrame:
    """Represents a single kinematic state.

    Positions are expressed in the world frame. The end-effector pose uses the
    (x, y, z, qx, qy, qz, qw) convention to remain simulator-agnostic.
    """

    timestamp: float
    joint_positions: Sequence[float]
    end_effector_pose: Sequence[float]
    base_frame: str = "world"

    def as_dict(self) -> Dict[str, object]:
        return {
            "timestamp": self.timestamp,
            "joint_positions": list(self.joint_positions),
            "end_effector_pose": list(self.end_effector_pose),
            "base_frame": self.base_frame,
        }


@dataclass
class KinematicTrajectory:
    """A time-ordered sequence of kinematic frames."""

    frames: List[KinematicFrame]
    metadata: TrajectoryMetadata

    def validate(self) -> None:
        """Checks basic invariants for downstream physical projection."""
        if not self.frames:
            raise ValueError("KinematicTrajectory contains no frames")
        timestamps = [frame.timestamp for frame in self.frames]
        if any(t2 < t1 for t1, t2 in zip(timestamps, timestamps[1:])):
            raise ValueError("Frame timestamps must be non-decreasing")
        ee_lengths = {len(frame.end_effector_pose) for frame in self.frames}
        if ee_lengths != {7}:
            raise ValueError("End-effector pose must contain 7 elements (x, y, z, qx, qy, qz, qw)")

    def as_dict(self) -> Dict[str, object]:
        self.validate()
        return {
            "metadata": self.metadata.as_dict(),
            "frames": [frame.as_dict() for frame in self.frames],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(self.as_dict(), fp, ensure_ascii=True, indent=2)

    @classmethod
    def from_iterable(cls, frames: Iterable[KinematicFrame], metadata: TrajectoryMetadata) -> "KinematicTrajectory":
        return cls(frames=list(frames), metadata=metadata)
