"""YAML loader for kinematic generator configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml  # type: ignore[import-untyped]

from .configs import (
    AugmentationConfig,
    DexCapConfig,
    KinematicGeneratorConfig,
    MimicGenConfig,
    R2R2RConfig,
)


def _load_augmentations(data: Mapping[str, Any] | None) -> AugmentationConfig:
    data = data or {}
    return AugmentationConfig(
        position_noise_m=float(data.get("position_noise_m", 0.01)),
        rotation_noise_deg=float(data.get("rotation_noise_deg", 2.0)),
        time_warp_factor=float(data.get("time_warp_factor", 0.05)),
    )


def _load_r2r2r(data: Mapping[str, Any] | None) -> Optional[R2R2RConfig]:
    if not data:
        return None
    return R2R2RConfig(
        capture_root=Path(data["capture_root"]),
        reconstructions_dir=(
            Path(data["reconstructions_dir"])
            if data.get("reconstructions_dir")
            else None
        ),
        clip_filter=list(data.get("clip_filter", [])),
        enable_gaussian_splatting=bool(data.get("enable_gaussian_splatting", True)),
        background_randomization=bool(data.get("background_randomization", True)),
        augmentations=_load_augmentations(data.get("augmentations")),
    )


def _load_mimicgen(data: Mapping[str, Any] | None) -> Optional[MimicGenConfig]:
    if not data:
        return None
    return MimicGenConfig(
        dataset_root=Path(data["dataset_root"]),
        augmentations=_load_augmentations(data.get("augmentations")),
        max_augmentations_per_demo=int(data.get("max_augmentations_per_demo", 8)),
        dexcap_mixture_ratio=float(data.get("dexcap_mixture_ratio", 0.25)),
    )


def _load_dexcap(data: Mapping[str, Any] | None) -> Optional[DexCapConfig]:
    if not data:
        return None
    return DexCapConfig(
        dataset_root=Path(data["dataset_root"]),
        clip_filter=list(data.get("clip_filter", [])),
        augmentations=_load_augmentations(data.get("augmentations")),
    )


def load_kinematic_generator_config(path: Path) -> KinematicGeneratorConfig:
    """Load KinematicGeneratorConfig from a YAML file."""
    with path.open("r", encoding="utf-8") as fp:
        raw: Dict[str, Any] = yaml.safe_load(fp)
    return KinematicGeneratorConfig(
        output_dir=Path(raw["output_dir"]),
        max_trajectories=raw.get("max_trajectories"),
        r2r2r=_load_r2r2r(raw.get("r2r2r")),
        mimicgen=_load_mimicgen(raw.get("mimicgen")),
        dexcap=_load_dexcap(raw.get("dexcap")),
    )


__all__ = ["load_kinematic_generator_config"]
