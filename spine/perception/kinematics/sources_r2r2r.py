"""Thin R2R2R adapter wrapper with dependency detection."""

from __future__ import annotations

import logging
from typing import Iterable

from .configs import R2R2RConfig
from .data_structures import KinematicTrajectory, TrajectoryMetadata
from .dependencies import require_dependency
from .parsing import discover_json_trajectories, parse_trajectory_json
from .sources import KinematicSourceAdapter
from .augmentations import maybe_augment

logger = logging.getLogger(__name__)


class R2R2RAdapter(KinematicSourceAdapter):
    """Adapter for R2R2R-generated kinematic skeletons."""

    def __init__(self, config: R2R2RConfig) -> None:
        super().__init__("r2r2r")
        self.config = config

    def _ensure_available(self) -> None:
        require_dependency(
            module_name="r2r2r",
            repo_name="Real2Render2Real",
            repo_url="https://github.com/uynitsuj/real2render2real",
            install_subdir="real2render2real",
        )

    def generate(self) -> Iterable[KinematicTrajectory]:
        self._ensure_available()
        json_paths = discover_json_trajectories(self.config.capture_root, self.config.clip_filter)
        if not json_paths:
            raise FileNotFoundError(f"{self.config.capture_root} 下未发现 R2R2R 轨迹 JSON")
        for path in json_paths:
            metadata = TrajectoryMetadata(
                source="r2r2r",
                clip_id=path.stem,
                augmentations=self.config.augmentations.as_dict(),
            )
            logger.debug("解析 R2R2R 轨迹 %s", path)
            yield from maybe_augment(parse_trajectory_json(path, metadata), self.config.augmentations, num_aug=1)


__all__ = ["R2R2RAdapter"]
