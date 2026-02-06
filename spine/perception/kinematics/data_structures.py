"""Lightweight data structures for kinematic trajectory IO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrajectoryMetadata:
    source: str | None = None
    clip_id: str | None = None
    notes: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        source: str | None = None,
        clip_id: str | None = None,
        notes: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        self.source = source
        self.clip_id = clip_id
        self.notes = {} if notes is None else dict(notes)
        self.extra = dict(kwargs)

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.source is not None:
            out["source"] = self.source
        if self.clip_id is not None:
            out["clip_id"] = self.clip_id
        out["notes"] = dict(self.notes)
        out.update(self.extra)
        return out


@dataclass
class KinematicFrame:
    timestamp: float
    joint_positions: list[float]
    end_effector_pose: list[float]
    base_frame: str = "world"


@dataclass
class KinematicTrajectory:
    frames: list[KinematicFrame]
    metadata: TrajectoryMetadata

