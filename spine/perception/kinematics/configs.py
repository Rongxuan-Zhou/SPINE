"""Configuration objects for kinematic skeleton generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence


@dataclass
class AugmentationConfig:
    """Controls lightweight kinematic augmentations applied to demonstrations.

    All units are explicit to keep downstream physical projection aligned with the
    same conventions used in the simulator: meters for translation, degrees for
    orientation jitter, and unitless time warp.
    """

    position_noise_m: float = 0.01
    rotation_noise_deg: float = 2.0
    time_warp_factor: float = 0.05

    def as_dict(self) -> Dict[str, float]:
        return {
            "position_noise_m": self.position_noise_m,
            "rotation_noise_deg": self.rotation_noise_deg,
            "time_warp_factor": self.time_warp_factor,
        }


@dataclass
class R2R2RConfig:
    """Configuration for R2R2R kinematic extraction."""

    capture_root: Path
    reconstructions_dir: Optional[Path] = None
    clip_filter: Sequence[str] = field(default_factory=list)
    enable_gaussian_splatting: bool = True
    background_randomization: bool = True
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)

    def as_dict(self) -> Dict[str, object]:
        return {
            "capture_root": str(self.capture_root),
            "reconstructions_dir": (
                str(self.reconstructions_dir) if self.reconstructions_dir else None
            ),
            "clip_filter": list(self.clip_filter),
            "enable_gaussian_splatting": self.enable_gaussian_splatting,
            "background_randomization": self.background_randomization,
            "augmentations": self.augmentations.as_dict(),
        }


@dataclass
class MimicGenConfig:
    """Configuration for MimicGen-generated trajectories."""

    dataset_root: Path
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    max_augmentations_per_demo: int = 8
    dexcap_mixture_ratio: float = 0.25

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset_root": str(self.dataset_root),
            "augmentations": self.augmentations.as_dict(),
            "max_augmentations_per_demo": self.max_augmentations_per_demo,
            "dexcap_mixture_ratio": self.dexcap_mixture_ratio,
        }


@dataclass
class DexCapConfig:
    """Configuration for DexCap demonstrations."""

    dataset_root: Path
    clip_filter: Sequence[str] = field(default_factory=list)
    augmentations: AugmentationConfig = field(default_factory=AugmentationConfig)
    use_ik: bool = False
    mjcf_path: Optional[Path] = None
    frame_dt: float = 0.033  # DexCap raw frame spacing (~30 Hz)
    ik_horizon: Optional[int] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset_root": str(self.dataset_root),
            "clip_filter": list(self.clip_filter),
            "augmentations": self.augmentations.as_dict(),
            "use_ik": self.use_ik,
            "mjcf_path": str(self.mjcf_path) if self.mjcf_path else None,
            "frame_dt": self.frame_dt,
            "ik_horizon": self.ik_horizon,
        }


@dataclass
class KinematicGeneratorConfig:
    """Top-level configuration for the kinematic generator."""

    output_dir: Path
    r2r2r: Optional[R2R2RConfig] = None
    mimicgen: Optional[MimicGenConfig] = None
    dexcap: Optional[DexCapConfig] = None
    max_trajectories: Optional[int] = None

    def active_sources(self) -> List[str]:
        """Returns names of configured sources in order."""
        names: List[str] = []
        if self.r2r2r:
            names.append("r2r2r")
        if self.mimicgen:
            names.append("mimicgen")
        if self.dexcap:
            names.append("dexcap")
        return names

    def as_dict(self) -> Dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "r2r2r": self.r2r2r.as_dict() if self.r2r2r else None,
            "mimicgen": self.mimicgen.as_dict() if self.mimicgen else None,
            "dexcap": self.dexcap.as_dict() if self.dexcap else None,
            "max_trajectories": self.max_trajectories,
        }
