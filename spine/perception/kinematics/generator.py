"""Orchestrates kinematic trajectory generation from multiple sources."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Sequence

from .data_structures import KinematicTrajectory
from .sources import KinematicSourceAdapter

logger = logging.getLogger(__name__)


class KinematicGenerator:
    """Runs configured kinematic generators and persists trajectories to disk."""

    def __init__(
        self,
        output_dir: Path,
        adapters: Sequence[KinematicSourceAdapter],
        max_trajectories: int | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.adapters: List[KinematicSourceAdapter] = list(adapters)
        self.max_trajectories = max_trajectories

    def run(self) -> List[Path]:
        """Executes all adapters and writes trajectories as JSON."""
        generated_paths: List[Path] = []
        for adapter in self.adapters:
            logger.info("Running kinematic source adapter: %s", adapter.source_name)
            for idx, trajectory in enumerate(adapter.generate()):
                if self.max_trajectories is not None and len(generated_paths) >= self.max_trajectories:
                    logger.info("Reached max_trajectories=%s; stopping early", self.max_trajectories)
                    return generated_paths
                file_path = self._write_trajectory(adapter, idx, trajectory)
                generated_paths.append(file_path)
        return generated_paths

    def _write_trajectory(self, adapter: KinematicSourceAdapter, index: int, trajectory: KinematicTrajectory) -> Path:
        trajectory.validate()
        adapter_dir = self.output_dir / adapter.source_name
        filename = self._build_filename(adapter, index, trajectory)
        path = adapter_dir / filename
        trajectory.write_json(path)
        logger.debug("Wrote kinematic trajectory to %s", path)
        return path

    @staticmethod
    def _build_filename(adapter: KinematicSourceAdapter, index: int, trajectory: KinematicTrajectory) -> str:
        clip = trajectory.metadata.clip_id or "clip"
        clip_sanitized = clip.replace(" ", "_")
        return f"{clip_sanitized}_{adapter.source_name}_{index:04d}.json"

    def register_adapter(self, adapter: KinematicSourceAdapter) -> None:
        self.adapters.append(adapter)
