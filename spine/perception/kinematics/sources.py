"""Adapters that wrap external kinematic data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from .data_structures import KinematicTrajectory


class KinematicSourceAdapter(ABC):
    """Base class for kinematic data sources."""

    source_name: str

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name

    @abstractmethod
    def generate(self) -> Iterable[KinematicTrajectory]:
        """Yield trajectories in the world frame for downstream physics infill."""
        raise NotImplementedError


# Per-source adapters live in dedicated modules to keep the core decoupled.
from .sources_dexcap import DexCapAdapter  # noqa: E402
from .sources_mimicgen import MimicGenAdapter  # noqa: E402
from .sources_r2r2r import R2R2RAdapter  # noqa: E402

__all__ = [
    "DexCapAdapter",
    "KinematicSourceAdapter",
    "MimicGenAdapter",
    "R2R2RAdapter",
]
